import collections
import io
import sys
import types
from typing import (
import torch
import torch.distributed.rpc as rpc
from torch import Tensor, device, dtype, nn
from torch.distributed.nn.jit import instantiator
from torch.distributed import _remote_device
from torch.distributed.rpc.internal import _internal_rpc_pickler
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
@staticmethod
def init_from_module_rref(remote_device: str, module_rref: rpc.RRef[nn.Module], _module_interface_cls: Any=None):
    """
        Besides the constructor, a RemoteModule instance can also be initialized given a module RRef.

        This alternate initialization method can be particularly useful if we want to create multiple
        RemoteModule instances that share the same underlying module and reduce memory consumption.

        Moreover, this also provides a workaround for passing script RemoteModule over RPC,
        which is not supported. The recommended way is as follows:

            1. the sender creates a RemoteModule;
            2. the sender sends its ``module_rref`` over RPC;
            3. the receiver calls this method to initialize another RemoteModule using the same ``module_rref``.

        Example::
            Run the following code in two different processes:

            >>> # xdoctest: +SKIP("distributed")
            >>> # On worker 0:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>> from torch import nn, Tensor
            >>> from torch.distributed.nn.api.remote_module import RemoteModule
            >>>
            >>> rpc.init_rpc("worker0", rank=0, world_size=2)
            >>> remote_module = RemoteModule(
            >>>     "worker1/cpu", nn.Linear, args=(20, 30),
            >>> )
            >>>
            >>> remote_module1 = rpc.rpc_sync(
            >>>     "worker1/cpu",
            >>>     RemoteModule.init_from_module_rref,
            >>>     ("worker1/cpu", remote_module1.get_module_rref()),
            >>> )
            >>> rpc.shutdown()

            >>> # On worker 1:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>>
            >>> rpc.init_rpc("worker1", rank=1, world_size=2)
            >>> rpc.shutdown()

        Args:
            remote_device (str): Device on the destination worker where we'd like to place this module.
                The device can be a local device or a remote device specified by one of the following remote
                formats:

                    1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                    2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").

                In addition, the device field can be optional and the default value is "cpu".
            module_rref (RRef[nn.Module]): The module reference shared by both the caller and
                the created remote module.
            _module_interface_cls (type, optional): The TorchScript interface type for the module
                to be created. The type object should be decorated by @torch.jit.interface.
                If not provided, the generated RemoteModule is not torchscript-able.
                Warning, this is an experimental API and susceptible to frequent changes.

        Returns:
            A remote module instance which wraps the :class:`~nn.Module` created by the
            user-provided ``module_rref``, it has a blocking ``forward`` method and an
            asynchronous ``forward_async`` method that returns a future of the ``forward`` call
            on the user-provided module on the remote side.
        """
    remote_module = object.__new__(RemoteModule)
    enable_moving_cpu_tensors_to_cuda = remote_module._prepare_init(remote_device)
    if _module_interface_cls is not None:
        remote_module.is_scriptable = True
        remote_module._init_template(_module_interface_cls, enable_moving_cpu_tensors_to_cuda)
    else:
        remote_module.is_scriptable = False
        remote_module.generated_methods = _NON_SCRIPTABLE_REMOTE_MODULE_MODULE._generated_methods
    remote_module.module_rref = module_rref
    remote_module._install_generated_methods()
    remote_module._check_attribute_picklability()
    return remote_module
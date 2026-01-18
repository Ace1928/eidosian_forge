from typing import Dict, List, Optional, Union
import torch
from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase
from . import constants as rpc_contants
def set_device_map(self, to: str, device_map: Dict[DeviceType, DeviceType]):
    """
        Set device mapping between each RPC caller and callee pair. This
        function can be called multiple times to incrementally add
        device placement configurations.

        Args:
            to (str): Callee name.
            device_map (Dict of int, str, or torch.device): Device placement
                mappings from this worker to the callee. This map must be
                invertible.

        Example:
            >>> # xdoctest: +SKIP("distributed")
            >>> # both workers
            >>> def add(x, y):
            >>>     print(x)  # tensor([1., 1.], device='cuda:1')
            >>>     return x + y, (x + y).to(2)
            >>>
            >>> # on worker 0
            >>> options = TensorPipeRpcBackendOptions(
            >>>     num_worker_threads=8,
            >>>     device_maps={"worker1": {0: 1}}
            >>>     # maps worker0's cuda:0 to worker1's cuda:1
            >>> )
            >>> options.set_device_map("worker1", {1: 2})
            >>> # maps worker0's cuda:1 to worker1's cuda:2
            >>>
            >>> rpc.init_rpc(
            >>>     "worker0",
            >>>     rank=0,
            >>>     world_size=2,
            >>>     backend=rpc.BackendType.TENSORPIPE,
            >>>     rpc_backend_options=options
            >>> )
            >>>
            >>> x = torch.ones(2)
            >>> rets = rpc.rpc_sync("worker1", add, args=(x.to(0), 1))
            >>> # The first argument will be moved to cuda:1 on worker1. When
            >>> # sending the return value back, it will follow the invert of
            >>> # the device map, and hence will be moved back to cuda:0 and
            >>> # cuda:1 on worker0
            >>> print(rets[0])  # tensor([2., 2.], device='cuda:0')
            >>> print(rets[1])  # tensor([2., 2.], device='cuda:1')
        """
    full_device_map = _to_device_map(device_map)
    curr_device_maps = super().device_maps
    if to in curr_device_maps:
        for k, v in full_device_map.items():
            if k in curr_device_maps[to] and v != curr_device_maps[to][k]:
                raise ValueError(f'`set_device_map` only supports 1-to-1 mapping, trying to map {k} to {v} and {curr_device_maps[to][k]}')
    super()._set_device_map(to, full_device_map)
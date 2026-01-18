from datetime import timedelta
import logging
import os
import threading
import warnings
from typing import Generator, Tuple
from urllib.parse import urlparse
import torch
import torch.distributed as dist
def init_rpc(name, backend=None, rank=-1, world_size=None, rpc_backend_options=None):
    """
        Initializes RPC primitives such as the local RPC agent
        and distributed autograd, which immediately makes the current
        process ready to send and receive RPCs.

        Args:
            name (str): a globally unique name of this node. (e.g.,
                ``Trainer3``, ``ParameterServer2``, ``Master``, ``Worker1``)
                Name can only contain number, alphabet, underscore, colon,
                and/or dash, and must be shorter than 128 characters.
            backend (BackendType, optional): The type of RPC backend
                implementation. Supported values is
                ``BackendType.TENSORPIPE`` (the default).
                See :ref:`rpc-backends` for more information.
            rank (int): a globally unique id/rank of this node.
            world_size (int): The number of workers in the group.
            rpc_backend_options (RpcBackendOptions, optional): The options
                passed to the RpcAgent constructor. It must be an agent-specific
                subclass of :class:`~torch.distributed.rpc.RpcBackendOptions`
                and contains agent-specific initialization configurations. By
                default, for all agents, it sets the default timeout to 60
                seconds and performs the rendezvous with an underlying process
                group initialized using ``init_method = "env://"``,
                meaning that environment variables ``MASTER_ADDR`` and
                ``MASTER_PORT`` need to be set properly. See
                :ref:`rpc-backends` for more information and find which options
                are available.
        """
    torch._C._log_api_usage_once('torch.distributed.init_rpc')
    if backend is not None and (not isinstance(backend, backend_registry.BackendType)):
        raise TypeError('Argument backend must be a member of BackendType')
    if rpc_backend_options is not None and (not isinstance(rpc_backend_options, RpcBackendOptions)):
        raise TypeError('Argument rpc_backend_options must be an instance of RpcBackendOptions')
    if backend is None and rpc_backend_options is not None:
        for candidate_backend in BackendType:
            if isinstance(rpc_backend_options, type(backend_registry.construct_rpc_backend_options(candidate_backend))):
                backend = candidate_backend
                break
        else:
            raise TypeError(f'Could not infer backend for options {rpc_backend_options}')
        if backend != BackendType.TENSORPIPE:
            logger.warning('RPC was initialized with no explicit backend but with options corresponding to %(backend)s, hence that backend will be used instead of the default BackendType.TENSORPIPE. To silence this warning pass `backend=%(backend)s` explicitly.', {'backend': backend})
    if backend is None:
        backend = BackendType.TENSORPIPE
    if rpc_backend_options is None:
        rpc_backend_options = backend_registry.construct_rpc_backend_options(backend)
    if not world_size:
        store = dist._create_store_from_options(rpc_backend_options, rank)
    else:
        global rendezvous_iterator
        rendezvous_iterator = dist.rendezvous(rpc_backend_options.init_method, rank=rank, world_size=world_size)
        store, _, _ = next(rendezvous_iterator)
    store.set_timeout(timedelta(seconds=rpc_backend_options.rpc_timeout))
    with _init_counter_lock:
        global _init_counter
        store = dist.PrefixStore(str(f'rpc_prefix_{_init_counter}'), store)
        _init_counter += 1
    dist_autograd._init(rank)
    _set_profiler_node_id(rank)
    _init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)
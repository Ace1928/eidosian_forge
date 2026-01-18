from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
def maybe_record_rpc_latency(state: '_channel._RPCState') -> None:
    """Record the latency of the RPC, if the plugin is registered and stats is enabled.

    This method will be called at the end of each RPC.

    Args:
    state: a grpc._channel._RPCState object which contains the stats related to the
    RPC.
    """
    for exclude_prefix in _SERVICES_TO_EXCLUDE:
        if exclude_prefix in state.method.encode('utf8'):
            return
    with get_plugin() as plugin:
        if not (plugin and plugin.stats_enabled):
            return
        rpc_latency_s = state.rpc_end_time - state.rpc_start_time
        rpc_latency_ms = rpc_latency_s * 1000
        plugin.record_rpc_latency(state.method, state.target, rpc_latency_ms, state.code)
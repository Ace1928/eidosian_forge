import pickle
from abc import ABC, abstractmethod
from types import LambdaType
from typing import Any, Callable, Dict
from uuid import uuid4
from triad import ParamDict, SerializableRLock, assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path, to_type
def make_rpc_server(conf: Any) -> RPCServer:
    """Make :class:`~.RPCServer` based on configuration.
    If '`fugue.rpc.server`` is set, then the value will be used as
    the server type for the initialization. Otherwise, a
    :class:`~.NativeRPCServer` instance will be returned

    :param conf: |FugueConfig|
    :return: the RPC server
    """
    conf = ParamDict(conf)
    tp = conf.get_or_none('fugue.rpc.server', str)
    t_server = NativeRPCServer if tp is None else to_type(tp, RPCServer)
    return t_server(conf)
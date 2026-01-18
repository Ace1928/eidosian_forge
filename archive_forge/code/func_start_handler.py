import pickle
from abc import ABC, abstractmethod
from types import LambdaType
from typing import Any, Callable, Dict
from uuid import uuid4
from triad import ParamDict, SerializableRLock, assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path, to_type
def start_handler(self) -> None:
    """Wrapper to start the server, do not override or call directly"""
    with self._rpchandler_lock:
        self.start_server()
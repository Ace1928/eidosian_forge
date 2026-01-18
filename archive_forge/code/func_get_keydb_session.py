from __future__ import annotations
import sys
import copy
import pathlib
import inspect
import functools
import importlib.util
from typing import Any, Dict, Callable, Union, Optional, Type, TypeVar, List, Tuple, cast, TYPE_CHECKING
from types import ModuleType
def get_keydb_session(name: Optional[str]='default', validate_active: Optional[bool]=None, **kwargs) -> Optional['KeyDBSession']:
    """
    Get the KeyDB Session
    """
    global _keydb_sessions, _keydb_enabled
    if not _keydb_sessions.get(name):
        try:
            from aiokeydb import KeyDBClient
            _keydb_sessions[name] = KeyDBClient.get_session(name=name, verbose=False, **kwargs)
        except Exception as e:
            from .logs import logger
            logger.warning('KeyDB is not available. Disabling')
            _keydb_enabled = False
            return None
    if validate_active and _keydb_enabled is None:
        try:
            _keydb_enabled = _keydb_sessions[name].ping()
        except Exception as e:
            from .logs import logger
            logger.warning(f'KeyDB Session {name} is not active: {e}')
            _keydb_enabled = False
    if validate_active and (not _keydb_enabled):
        return None
    return _keydb_sessions[name]
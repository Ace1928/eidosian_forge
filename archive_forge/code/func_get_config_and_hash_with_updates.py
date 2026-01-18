import contextlib
import copy
import hashlib
import inspect
import io
import pickle
import tokenize
import unittest
import warnings
from types import FunctionType, ModuleType
from typing import Any, Dict, Optional, Set, Tuple, Union
from unittest import mock
def get_config_and_hash_with_updates(self, updates: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    """Hashes the configs that are not compile_ignored, along with updates"""
    if any((k in self._compile_ignored_keys for k in updates)):
        raise ValueError('update keys cannot be @compile_ignored')
    cfg = {k: v for k, v in self._config.items() if k not in self._compile_ignored_keys}
    cfg.update(updates)
    hashed = self._get_hash(cfg)
    return (cfg, hashed)
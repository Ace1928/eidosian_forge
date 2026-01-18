from __future__ import annotations
import io
import logging
import os
import threading
import warnings
import weakref
from errno import ESPIPE
from glob import has_magic
from hashlib import sha256
from typing import ClassVar
from .callbacks import DEFAULT_CALLBACK
from .config import apply_config, conf
from .dircache import DirCache
from .transaction import Transaction
from .utils import (
def end_transaction(self):
    """Finish write transaction, non-context version"""
    self.transaction.complete()
    self._transaction = None
    for path in self._invalidated_caches_in_transaction:
        self.invalidate_cache(path)
    self._invalidated_caches_in_transaction.clear()
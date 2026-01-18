from __future__ import annotations
import contextlib
import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import local
from typing import TYPE_CHECKING, Any, ClassVar
from weakref import WeakValueDictionary
from ._error import Timeout
@property
def lock_file(self) -> str:
    """:return: path to the lock file"""
    return self._context.lock_file
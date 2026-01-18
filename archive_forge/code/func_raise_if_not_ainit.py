from __future__ import annotations
import os
import abc
import kr8s
import httpx
import base64
import functools
import contextlib
from typing import Optional, Dict, Any, List, Union, Generator, Literal, Type, AsyncGenerator, overload, TYPE_CHECKING
from .types import objects, aobjects
def raise_if_not_ainit(self):
    """
        Raises if the async client is not initialized
        """
    if not self._ainitialized:
        raise ValueError('The async client is not initialized')
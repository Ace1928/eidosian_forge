from __future__ import annotations  # isort: split
import __future__  # Regular import, not special!
import enum
import functools
import importlib
import inspect
import json
import socket as stdlib_socket
import sys
import types
from pathlib import Path, PurePath
from types import ModuleType
from typing import TYPE_CHECKING, Protocol
import attrs
import pytest
import trio
import trio.testing
from trio._tests.pytest_plugin import skip_if_optional_else_raise
from .. import _core, _util
from .._core._tests.tutil import slow
from .pytest_plugin import RUN_SLOW
def test_classes_are_final() -> None:
    assert not class_is_final(object)
    assert class_is_final(bool)
    for module in PUBLIC_MODULES:
        for name, class_ in module.__dict__.items():
            if not isinstance(class_, type):
                continue
            if name.startswith('_'):
                continue
            if inspect.isabstract(class_):
                continue
            if Protocol in class_.__bases__ or Protocol_ext in class_.__bases__:
                continue
            if issubclass(class_, BaseException):
                continue
            if class_ is trio.abc.Instrument or class_ is trio.socket.SocketType:
                continue
            if class_ is trio.Path:
                continue
            if name.endswith('Statistics'):
                continue
            assert class_is_final(class_)
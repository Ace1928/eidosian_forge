import abc
import collections
import contextlib
import io
from io import UnsupportedOperation
import os
import sys
from tempfile import TemporaryFile
from types import TracebackType
from typing import Any
from typing import AnyStr
from typing import BinaryIO
from typing import Final
from typing import final
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import SubRequest
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.reports import CollectReport
@contextlib.contextmanager
def global_and_fixture_disabled(self) -> Generator[None, None, None]:
    """Context manager to temporarily disable global and current fixture capturing."""
    do_fixture = self._capture_fixture and self._capture_fixture._is_started()
    if do_fixture:
        self.suspend_fixture()
    do_global = self._global_capturing and self._global_capturing.is_started()
    if do_global:
        self.suspend_global_capture()
    try:
        yield
    finally:
        if do_global:
            self.resume_global_capture()
        if do_fixture:
            self.resume_fixture()
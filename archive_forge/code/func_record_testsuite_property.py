from datetime import datetime
import functools
import os
import platform
import re
from typing import Callable
from typing import Dict
from typing import List
from typing import Match
from typing import Optional
from typing import Tuple
from typing import Union
import xml.etree.ElementTree as ET
from _pytest import nodes
from _pytest import timing
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprFileLocation
from _pytest.config import Config
from _pytest.config import filename_arg
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
import pytest
@pytest.fixture(scope='session')
def record_testsuite_property(request: FixtureRequest) -> Callable[[str, object], None]:
    """Record a new ``<property>`` tag as child of the root ``<testsuite>``.

    This is suitable to writing global information regarding the entire test
    suite, and is compatible with ``xunit2`` JUnit family.

    This is a ``session``-scoped fixture which is called with ``(name, value)``. Example:

    .. code-block:: python

        def test_foo(record_testsuite_property):
            record_testsuite_property("ARCH", "PPC")
            record_testsuite_property("STORAGE_TYPE", "CEPH")

    :param name:
        The property name.
    :param value:
        The property value. Will be converted to a string.

    .. warning::

        Currently this fixture **does not work** with the
        `pytest-xdist <https://github.com/pytest-dev/pytest-xdist>`__ plugin. See
        :issue:`7767` for details.
    """
    __tracebackhide__ = True

    def record_func(name: str, value: object) -> None:
        """No-op function in case --junit-xml was not passed in the command-line."""
        __tracebackhide__ = True
        _check_record_param_type('name', name)
    xml = request.config.stash.get(xml_key, None)
    if xml is not None:
        record_func = xml.add_global_property
    return record_func
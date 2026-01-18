from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    from sqlalchemy.testing import asyncio
    asyncio._maybe_async(plugin_base.after_test, item)
    yield
    global _current_class, _current_report
    if _current_class is not None and (nextitem is None or nextitem.getparent(pytest.Class) is not _current_class):
        _current_class = None
        try:
            asyncio._maybe_async_provisioning(plugin_base.stop_test_class_outside_fixtures, item.cls)
        except Exception as e:
            if _current_report.failed:
                if not e.args:
                    e.args = ('__Original test failure__:\n' + _current_report.longreprtext,)
                elif e.args[-1] and isinstance(e.args[-1], str):
                    args = list(e.args)
                    args[-1] += '\n__Original test failure__:\n' + _current_report.longreprtext
                    e.args = tuple(args)
                else:
                    e.args += ('__Original test failure__', _current_report.longreprtext)
            raise
        finally:
            _current_report = None
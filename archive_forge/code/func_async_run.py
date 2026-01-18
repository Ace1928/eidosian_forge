import unittest
import logging
import pytest
import sys
from functools import partial
import os
import threading
from kivy.graphics.cgl import cgl_get_backend_name
from kivy.input.motionevent import MotionEvent
def async_run(func=None, app_cls_func=None):

    def inner_func(func):
        if 'mock' == cgl_get_backend_name():
            return pytest.mark.skip(reason='Skipping because gl backend is set to mock')(func)
        if sys.version_info[0] < 3 or sys.version_info[1] <= 5:
            return pytest.mark.skip(reason='Skipping because graphics tests are not supported on py3.5, only on py3.6+')(func)
        if app_cls_func is not None:
            func = pytest.mark.parametrize('kivy_app', [[app_cls_func]], indirect=True)(func)
        if kivy_eventloop == 'asyncio':
            try:
                import pytest_asyncio
                return pytest.mark.asyncio(pytest_asyncio.fixture(func))
            except ImportError:
                return pytest.mark.skip(reason='KIVY_EVENTLOOP == "asyncio" but "pytest-asyncio" is not installed')(func)
        elif kivy_eventloop == 'trio':
            try:
                import trio
                from pytest_trio import trio_fixture
                func._force_trio_fixture = True
                return func
            except ImportError:
                return pytest.mark.skip(reason='KIVY_EVENTLOOP == "trio" but "pytest-trio" is not installed')(func)
        else:
            return pytest.mark.skip(reason='KIVY_EVENTLOOP must be set to either of "asyncio" or "trio" to run async tests')(func)
    if func is None:
        return inner_func
    return inner_func(func)
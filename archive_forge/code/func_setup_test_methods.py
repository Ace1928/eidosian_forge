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
@pytest.fixture(scope='function')
def setup_test_methods(request):
    from sqlalchemy.testing import asyncio
    self = request.instance
    if hasattr(self, 'setup_test'):
        asyncio._maybe_async(self.setup_test)
    if hasattr(self, 'setUp'):
        asyncio._maybe_async(self.setUp)
    yield
    asyncio._maybe_async(plugin_base.after_test_fixtures, self)
    if hasattr(self, 'tearDown'):
        asyncio._maybe_async(self.tearDown)
    if hasattr(self, 'teardown_test'):
        asyncio._maybe_async(self.teardown_test)
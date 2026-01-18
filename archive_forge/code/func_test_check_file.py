import asyncio
import os
import pathlib
import tempfile
from panel.io.location import Location
from panel.io.reload import (
from panel.io.state import state
from panel.tests.util import async_wait_until
def test_check_file():
    modify_times = {}
    _check_file(__file__, modify_times)
    assert modify_times[__file__] == os.stat(__file__).st_mtime
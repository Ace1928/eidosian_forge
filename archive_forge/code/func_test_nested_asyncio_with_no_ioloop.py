import asyncio
from unittest.mock import MagicMock
import pytest
import tornado
from nbclient.util import run_hook, run_sync
def test_nested_asyncio_with_no_ioloop():
    asyncio.set_event_loop(None)
    assert some_async_function() == 42
from __future__ import annotations
import io
import json
import logging
from typing import Any, Callable
import pytest
from jupyter_events import EventLogger
@pytest.fixture
def jp_event_handler(jp_event_sink: io.StringIO) -> logging.Handler:
    """A logging handler that captures any events emitted by the event handler"""
    return logging.StreamHandler(jp_event_sink)
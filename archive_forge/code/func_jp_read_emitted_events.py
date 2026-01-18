from __future__ import annotations
import io
import json
import logging
from typing import Any, Callable
import pytest
from jupyter_events import EventLogger
@pytest.fixture
def jp_read_emitted_events(jp_event_handler: logging.Handler, jp_event_sink: io.StringIO) -> Callable[..., list[str] | None]:
    """Reads list of events since last time it was called."""

    def _read() -> list[str] | None:
        jp_event_handler.flush()
        event_buf = jp_event_sink.getvalue().strip()
        output = [json.loads(item) for item in event_buf.split('\n')] if event_buf else None
        jp_event_sink.truncate(0)
        jp_event_sink.seek(0)
        return output
    return _read
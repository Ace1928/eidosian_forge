import logging
import pathlib
import json
import random
import string
import socket
import os
import threading
from typing import Dict, Optional
from datetime import datetime
from google.protobuf.json_format import Parse
from ray.core.generated.event_pb2 import Event
from ray._private.protobuf_compat import message_to_dict
def filter_event_by_level(event: Event, filter_event_level: str) -> bool:
    """Filter an event based on event level.

    Args:
        event: The event to filter.
        filter_event_level: The event level string to filter by. Any events
            that are lower than this level will be filtered.

    Returns:
        True if the event should be filtered, else False.
    """
    event_levels = {Event.Severity.TRACE: 0, Event.Severity.DEBUG: 1, Event.Severity.INFO: 2, Event.Severity.WARNING: 3, Event.Severity.ERROR: 4, Event.Severity.FATAL: 5}
    filter_event_level = filter_event_level.upper()
    filter_event_level = Event.Severity.Value(filter_event_level)
    if event_levels[event.severity] < event_levels[filter_event_level]:
        return True
    return False
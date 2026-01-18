from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def populate_confidence(self, metadata: Metadata) -> None:
    """Populate test result confidence using the provided metadata."""
    for message in self.messages:
        if message.confidence is None:
            message.confidence = calculate_confidence(message.path, message.line, metadata)
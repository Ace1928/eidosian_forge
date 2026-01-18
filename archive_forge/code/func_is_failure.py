from __future__ import annotations
import abc
import dataclasses
import datetime
import decimal
from xml.dom import minidom
from xml.etree import ElementTree as ET
@property
def is_failure(self) -> bool:
    """True if the test case contains failure info."""
    return bool(self.failures)
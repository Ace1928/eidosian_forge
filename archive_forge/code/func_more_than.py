from __future__ import annotations
from calendar import timegm
from os.path import splitext
from time import altzone, gmtime, localtime, time, timezone
from typing import (
from urllib.parse import quote, urlsplit, urlunsplit
import rdflib.graph  # avoid circular dependency
import rdflib.namespace
import rdflib.term
from rdflib.compat import sign
def more_than(sequence: Iterable[Any], number: int) -> int:
    """Returns 1 if sequence has more items than number and 0 if not."""
    i = 0
    for item in sequence:
        i += 1
        if i > number:
            return 1
    return 0
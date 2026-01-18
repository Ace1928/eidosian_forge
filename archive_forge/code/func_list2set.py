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
def list2set(seq: Iterable[_HashableT]) -> List[_HashableT]:
    """
    Return a new list without duplicates.
    Preserves the order, unlike set(seq)
    """
    seen = set()
    return [x for x in seq if x not in seen and (not seen.add(x))]
from __future__ import annotations
import xml.etree.ElementTree
import pytest
import dask.array as da
from dask.array.svg import draw_sizes
def parses(text):
    cleaned = text.replace('&rarr;', '')
    assert xml.etree.ElementTree.fromstring(cleaned) is not None
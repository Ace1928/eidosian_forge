from __future__ import annotations
import collections
from functools import partial
from typing import Any, Set
import html_text
import lxml.etree
from lxml.html.clean import Cleaner
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
def get_docid(self, node, itemids):
    return itemids[node]
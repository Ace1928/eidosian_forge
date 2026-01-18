from __future__ import annotations
import re
from markdown.treeprocessors import Treeprocessor, isString
from markdown.extensions import Extension
from typing import TYPE_CHECKING
def attributeCallback(match: re.Match[str]):
    el.set(match.group(1), match.group(2).replace('\n', ' '))
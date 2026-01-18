from __future__ import annotations
import copy
import typing as t
from . import common
def start_gtml_tab(self, attrs: dict[str, str]) -> None:
    if attrs.get('title', '').strip():
        self.hierarchy.append(attrs['title'].strip())
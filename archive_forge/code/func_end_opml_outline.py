from __future__ import annotations
import copy
from . import common, dates
def end_opml_outline(self) -> None:
    self.hierarchy.pop()
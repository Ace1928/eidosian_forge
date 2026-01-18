from __future__ import annotations
import copy
from . import common, dates
def end_opml_title(self) -> None:
    value = self.get_text()
    if value:
        self.harvest['meta']['title'] = value
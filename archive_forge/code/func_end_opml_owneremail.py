from __future__ import annotations
import copy
from . import common, dates
def end_opml_owneremail(self) -> None:
    value = self.get_text()
    if value:
        self.harvest['meta'].setdefault('author', common.SuperDict())
        self.harvest['meta']['author']['email'] = value
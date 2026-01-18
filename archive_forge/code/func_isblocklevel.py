from __future__ import annotations
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from . import util
import re
def isblocklevel(self, html: str) -> bool:
    """ Check is block of HTML is block-level. """
    m = self.BLOCK_LEVEL_REGEX.match(html)
    if m:
        if m.group(1)[0] in ('!', '?', '@', '%'):
            return True
        return self.md.is_block_level(m.group(1))
    return False
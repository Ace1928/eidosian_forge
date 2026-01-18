import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
@property
def shared_atts(self) -> Dict[str, Union[int, bool]]:
    """Gets atts shared among all nonzero length component Chunks"""
    atts = {}
    first = self.chunks[0]
    for att in sorted(first.atts):
        if all((fs.atts.get(att, '???') == first.atts[att] for fs in self.chunks if len(fs) > 0)):
            atts[att] = first.atts[att]
    return atts
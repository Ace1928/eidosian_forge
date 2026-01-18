from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_eprint(self, e):
    return href[join['https://arxiv.org/abs/', field('eprint', raw=True)], join['arXiv:', field('eprint', raw=True)]]
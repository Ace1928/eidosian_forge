from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_web_refs(self, e):
    return sentence[optional[self.format_url(e), optional[' (visited on ', field('urldate'), ')']], optional[self.format_eprint(e)], optional[self.format_pubmed(e)], optional[self.format_doi(e)]]
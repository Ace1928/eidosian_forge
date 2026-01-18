from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_pubmed(self, e):
    return href[join['https://www.ncbi.nlm.nih.gov/pubmed/', field('pubmed', raw=True)], join['PMID:', field('pubmed', raw=True)]]
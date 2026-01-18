from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_chapter_and_pages(self, e):
    return join(sep=', ')[optional[together['chapter', field('chapter')]], optional[together['pages', pages]]]
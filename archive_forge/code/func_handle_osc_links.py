import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def handle_osc_links(self, part: OSC_Link) -> str:
    if self.latex:
        self.hyperref = True
        return '\\href{%s}{%s}' % (part.url, part.text)
    return '<a href="%s">%s</a>' % (part.url, part.text)
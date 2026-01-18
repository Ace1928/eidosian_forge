import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def do_linkify(self, line: str) -> str:
    if not isinstance(line, str):
        return line
    if self.latex:
        return self.url_matcher.sub('\\\\url{\\1}', line)
    return self.url_matcher.sub('<a href="\\1">\\1</a>', line)
import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def produce_headers(self) -> str:
    return '<style type="text/css">\n%(style)s\n</style>\n' % {'style': '\n'.join(map(str, get_styles(self.dark_bg, self.line_wrap, self.scheme)))}
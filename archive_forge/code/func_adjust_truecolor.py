import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def adjust_truecolor(self, ansi_code: int, r: int, g: int, b: int) -> None:
    parameter = '{:03d}{:03d}{:03d}'.format(r, g, b)
    is_foreground = ansi_code == ANSI_FOREGROUND
    add_truecolor_style_rule(is_foreground, ansi_code, r, g, b, parameter)
    if is_foreground:
        self.foreground = (ansi_code, parameter)
    else:
        self.background = (ansi_code, parameter)
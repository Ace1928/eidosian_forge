from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def describe_probes():
    """Display a (tabular) description of all available probes in Z3."""
    if in_html_mode():
        even = True
        print('<table border="1" cellpadding="2" cellspacing="0">')
        for p in probes():
            if even:
                print('<tr style="background-color:#CFCFCF">')
                even = False
            else:
                print('<tr>')
                even = True
            print('<td>%s</td><td>%s</td></tr>' % (p, insert_line_breaks(probe_description(p), 40)))
        print('</table>')
    else:
        for p in probes():
            print('%s : %s' % (p, probe_description(p)))
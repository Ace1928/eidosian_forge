import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
def script_from_examples(s):
    """Extract script from text with examples.

       Converts text with examples to a Python script.  Example input is
       converted to regular code.  Example output and all other words
       are converted to comments:

       >>> text = '''
       ...       Here are examples of simple math.
       ...
       ...           Python has super accurate integer addition
       ...
       ...           >>> 2 + 2
       ...           5
       ...
       ...           And very friendly error messages:
       ...
       ...           >>> 1/0
       ...           To Infinity
       ...           And
       ...           Beyond
       ...
       ...           You can use logic if you want:
       ...
       ...           >>> if 0:
       ...           ...    blah
       ...           ...    blah
       ...           ...
       ...
       ...           Ho hum
       ...           '''

       >>> print(script_from_examples(text))
       # Here are examples of simple math.
       #
       #     Python has super accurate integer addition
       #
       2 + 2
       # Expected:
       ## 5
       #
       #     And very friendly error messages:
       #
       1/0
       # Expected:
       ## To Infinity
       ## And
       ## Beyond
       #
       #     You can use logic if you want:
       #
       if 0:
          blah
          blah
       #
       #     Ho hum
       <BLANKLINE>
       """
    output = []
    for piece in DocTestParser().parse(s):
        if isinstance(piece, Example):
            output.append(piece.source[:-1])
            want = piece.want
            if want:
                output.append('# Expected:')
                output += ['## ' + l for l in want.split('\n')[:-1]]
        else:
            output += [_comment_line(l) for l in piece.split('\n')[:-1]]
    while output and output[-1] == '#':
        output.pop()
    while output and output[0] == '#':
        output.pop(0)
    return '\n'.join(output) + '\n'
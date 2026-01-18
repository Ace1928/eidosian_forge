import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
def create_assignment(self, target, expression):
    template = '\n      target = expression\n    '
    return templates.replace(template, target=target, expression=expression)
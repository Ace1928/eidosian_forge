import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
def debug_print(self, node):
    """Helper method useful for debugging. Prints the AST."""
    if __debug__:
        print(pretty_printer.fmt(node))
    return node
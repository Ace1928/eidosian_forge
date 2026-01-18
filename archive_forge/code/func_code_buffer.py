import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
@property
def code_buffer(self):
    return self._output_code
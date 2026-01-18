import collections
import difflib
import io
import os
import tokenize
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.util import tf_inspect
def as_frame(self):
    """Returns a 4-tuple consistent with the return of traceback.extract_tb."""
    return (self.loc.filename, self.loc.lineno, self.function_name, self.source_code_line)
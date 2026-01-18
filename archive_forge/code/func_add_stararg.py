import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.utils import ag_logging
def add_stararg(self, a):
    self._consume_args()
    self._argspec.append(gast.Call(gast.Name('tuple', ctx=gast.Load(), annotation=None, type_comment=None), args=[a], keywords=()))
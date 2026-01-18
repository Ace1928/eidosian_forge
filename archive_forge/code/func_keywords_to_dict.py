import ast
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
def keywords_to_dict(keywords):
    """Converts a list of ast.keyword objects to a dict."""
    keys = []
    values = []
    for kw in keywords:
        keys.append(gast.Constant(kw.arg, kind=None))
        values.append(kw.value)
    return gast.Dict(keys=keys, values=values)
import itertools
from typing import Any, Callable, Dict, Set
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
def res_call(self, ns, types_ns, node, f_type, args, keywords):
    """Resolves the return type an external function or method call.

    Args:
      ns: namespace
      types_ns: types namespace
      node: str, the function name
      f_type: types of the actual function being called, if known
      args: types of each respective argument in node.args
      keywords: types of each respective argument in node.keywords

    Returns:
      Tuple (return_type, side_effect_types). The first element is just the
      return types of the function. The second element is a map from
      argument names to sets of types, and allow modelling side effects of
      functions (for example via global or nonlocal).
    """
    raise NotImplementedError('subclasses must implement')
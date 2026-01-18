import itertools
from typing import Any, Callable, Dict, Set
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
def res_list_literal(self, ns, elt_types):
    """Resolves the type of a list literal from its elements."""
    raise NotImplementedError('subclasses must implement')
import collections
import dataclasses
import functools
import io
import itertools
import threading
from absl import app
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.util import nest
def show_backref(target, max_depth=3):
    """Returns a dot graph of all the objects that are referencing the target.

  A object referencing graph is useful to debug memory leak like circular
  reference. objgraph provides a good visualization of the memory graph than
  most python built-in utilities like gc.get_referrers(), which are not
  human-readable sometimes.

  The dot graph will be written to a string IO object, and can be rendered with
  graphviz in operating system.
  E.g. dot -Tpng {$dot_graph} -o output.png
  Args:
    target: The target object for the memory graph.
    max_depth: The maximum depth of the graph. By default 3 layers of references
      are used. Increases this a lot may result in the graph growing too big.

  Returns:
    A string that contains the object reference graph.
  Raises:
    NotImplementedError: if objgraph is not installed.
  """
    if objgraph is None:
        raise NotImplementedError('objgraph is not installed.')
    string_io = io.StringIO()
    objgraph.show_backrefs(target, max_depth=max_depth, output=string_io)
    graph = string_io.getvalue()
    string_io.close()
    return graph
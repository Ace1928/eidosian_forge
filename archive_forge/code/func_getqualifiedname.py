import builtins
import inspect
import itertools
import linecache
import sys
import threading
import types
from tensorflow.python.util import tf_inspect
def getqualifiedname(namespace, object_, max_depth=5, visited=None):
    """Returns the name by which a value can be referred to in a given namespace.

  If the object defines a parent module, the function attempts to use it to
  locate the object.

  This function will recurse inside modules, but it will not search objects for
  attributes. The recursion depth is controlled by max_depth.

  Args:
    namespace: Dict[str, Any], the namespace to search into.
    object_: Any, the value to search.
    max_depth: Optional[int], a limit to the recursion depth when searching
      inside modules.
    visited: Optional[Set[int]], ID of modules to avoid visiting.
  Returns: Union[str, None], the fully-qualified name that resolves to the value
    o, or None if it couldn't be found.
  """
    if visited is None:
        visited = set()
    namespace = dict(namespace)
    for name in namespace:
        if object_ is namespace[name]:
            return name
    parent = tf_inspect.getmodule(object_)
    if parent is not None and parent is not object_ and (parent is not namespace):
        parent_name = getqualifiedname(namespace, parent, max_depth=0, visited=visited)
        if parent_name is not None:
            name_in_parent = getqualifiedname(parent.__dict__, object_, max_depth=0, visited=visited)
            assert name_in_parent is not None, 'An object should always be found in its owner module'
            return '{}.{}'.format(parent_name, name_in_parent)
    if max_depth:
        for name in namespace.keys():
            value = namespace[name]
            if tf_inspect.ismodule(value) and id(value) not in visited:
                visited.add(id(value))
                name_in_module = getqualifiedname(value.__dict__, object_, max_depth - 1, visited)
                if name_in_module is not None:
                    return '{}.{}'.format(name, name_in_module)
    return None
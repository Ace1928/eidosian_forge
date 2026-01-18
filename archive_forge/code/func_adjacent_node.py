import inspect
import re
import six
def adjacent_node(name):
    """
            Returns an adjacent node or ourself.
            """
    if name == self._path_current:
        return self
    elif name == self._path_previous:
        if self._parent is not None:
            return self._parent
        else:
            return self
    else:
        return self.get_child(name)
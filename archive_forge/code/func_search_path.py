import types
import weakref
import six
from apitools.base.protorpclite import util
def search_path():
    """Performs a single iteration searching the path from relative_to.

        This is the function that searches up the path from a relative object.

          fully.qualified.object . relative.or.nested.Definition
                                   ---------------------------->
                                                      ^
                                                      |
                                this part of search --+

        Returns:
          Message or Enum at the end of name_path, else None.
        """
    next_part = relative_to
    for node in name_path:
        attribute = getattr(next_part, node, None)
        if attribute is not None:
            next_part = attribute
        elif next_part is None or isinstance(next_part, types.ModuleType):
            if next_part is None:
                module_name = node
            else:
                module_name = '%s.%s' % (next_part.__name__, node)
            try:
                fromitem = module_name.split('.')[-1]
                next_part = importer(module_name, '', '', [str(fromitem)])
            except ImportError:
                return None
        else:
            return None
        if not isinstance(next_part, types.ModuleType):
            if not (isinstance(next_part, type) and issubclass(next_part, (Message, Enum))):
                return None
    return next_part
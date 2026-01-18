import types
import weakref
import six
from apitools.base.protorpclite import util
@util.positional(2)
def find_definition(name, relative_to=None, importer=__import__):
    """Find definition by name in module-space.

    The find algorthm will look for definitions by name relative to a
    message definition or by fully qualfied name. If no definition is
    found relative to the relative_to parameter it will do the same
    search against the container of relative_to. If relative_to is a
    nested Message, it will search its message_definition(). If that
    message has no message_definition() it will search its module. If
    relative_to is a module, it will attempt to look for the
    containing module and search relative to it. If the module is a
    top-level module, it will look for the a message using a fully
    qualified name. If no message is found then, the search fails and
    DefinitionNotFoundError is raised.

    For example, when looking for any definition 'foo.bar.ADefinition'
    relative to an actual message definition abc.xyz.SomeMessage:

      find_definition('foo.bar.ADefinition', SomeMessage)

    It is like looking for the following fully qualified names:

      abc.xyz.SomeMessage. foo.bar.ADefinition
      abc.xyz. foo.bar.ADefinition
      abc. foo.bar.ADefinition
      foo.bar.ADefinition

    When resolving the name relative to Message definitions and modules, the
    algorithm searches any Messages or sub-modules found in its path.
    Non-Message values are not searched.

    A name that begins with '.' is considered to be a fully qualified
    name. The name is always searched for from the topmost package.
    For example, assume two message types:

      abc.xyz.SomeMessage
      xyz.SomeMessage

    Searching for '.xyz.SomeMessage' relative to 'abc' will resolve to
    'xyz.SomeMessage' and not 'abc.xyz.SomeMessage'.  For this kind of name,
    the relative_to parameter is effectively ignored and always set to None.

    For more information about package name resolution, please see:

      http://code.google.com/apis/protocolbuffers/docs/proto.html#packages

    Args:
      name: Name of definition to find.  May be fully qualified or relative
        name.
      relative_to: Search for definition relative to message definition or
        module. None will cause a fully qualified name search.
      importer: Import function to use for resolving modules.

    Returns:
      Enum or Message class definition associated with name.

    Raises:
      DefinitionNotFoundError if no definition is found in any search path.

    """
    if not (relative_to is None or isinstance(relative_to, types.ModuleType) or (isinstance(relative_to, type) and issubclass(relative_to, Message))):
        raise TypeError('relative_to must be None, Message definition or module.  Found: %s' % relative_to)
    name_path = name.split('.')
    if not name_path[0]:
        relative_to = None
        name_path = name_path[1:]

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
    while True:
        found = search_path()
        if isinstance(found, type) and issubclass(found, (Enum, Message)):
            return found
        elif relative_to is None:
            raise DefinitionNotFoundError('Could not find definition for %s' % name)
        elif isinstance(relative_to, types.ModuleType):
            module_path = relative_to.__name__.split('.')[:-1]
            if not module_path:
                relative_to = None
            else:
                relative_to = importer('.'.join(module_path), '', '', [module_path[-1]])
        elif isinstance(relative_to, type) and issubclass(relative_to, Message):
            parent = relative_to.message_definition()
            if parent is None:
                last_module_name = relative_to.__module__.split('.')[-1]
                relative_to = importer(relative_to.__module__, '', '', [last_module_name])
            else:
                relative_to = parent
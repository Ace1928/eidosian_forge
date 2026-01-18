import sys
import traceback
def import_any(module, *modules):
    """Try to import a module from a list of modules.

    :param modules: A list of modules to try and import
    :returns: The first module found that can be imported
    :raises ImportError: If no modules can be imported from list

    .. versionadded:: 3.8
    """
    for module_name in (module,) + modules:
        imported_module = try_import(module_name)
        if imported_module:
            return imported_module
    raise ImportError('Unable to import any modules from the list %s' % str(modules))
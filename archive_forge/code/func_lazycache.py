import functools
import sys
import os
import tokenize
def lazycache(filename, module_globals):
    """Seed the cache for filename with module_globals.

    The module loader will be asked for the source only when getlines is
    called, not immediately.

    If there is an entry in the cache already, it is not altered.

    :return: True if a lazy load is registered in the cache,
        otherwise False. To register such a load a module loader with a
        get_source method must be found, the filename must be a cacheable
        filename, and the filename must not be already cached.
    """
    if filename in cache:
        if len(cache[filename]) == 1:
            return True
        else:
            return False
    if not filename or (filename.startswith('<') and filename.endswith('>')):
        return False
    if module_globals and '__name__' in module_globals:
        name = module_globals['__name__']
        if (loader := module_globals.get('__loader__')) is None:
            if (spec := module_globals.get('__spec__')):
                try:
                    loader = spec.loader
                except AttributeError:
                    pass
        get_source = getattr(loader, 'get_source', None)
        if name and get_source:
            get_lines = functools.partial(get_source, name)
            cache[filename] = (get_lines,)
            return True
    return False
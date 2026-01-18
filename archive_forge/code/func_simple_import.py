def simple_import(s):
    """
    Import a module, or import an object from a module.

    A name like ``foo.bar.baz`` can be a module ``foo.bar.baz`` or a
    module ``foo.bar`` with an object ``baz`` in it, or a module
    ``foo`` with an object ``bar`` with an attribute ``baz``.
    """
    parts = s.split('.')
    module = import_module(parts[0])
    name = parts[0]
    parts = parts[1:]
    last_import_error = None
    while parts:
        name += '.' + parts[0]
        try:
            module = import_module(name)
            parts = parts[1:]
        except ImportError as e:
            last_import_error = e
            break
    obj = module
    while parts:
        try:
            obj = getattr(module, parts[0])
        except AttributeError:
            raise ImportError('Cannot find %s in module %r (stopped importing modules with error %s)' % (parts[0], module, last_import_error))
        parts = parts[1:]
    return obj
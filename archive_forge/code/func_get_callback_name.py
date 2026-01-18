import inspect
def get_callback_name(cb):
    """Tries to get a callbacks fully-qualified name.

    If no name can be produced ``repr(cb)`` is called and returned.
    """
    segments = []
    try:
        segments.append(cb.__qualname__)
    except AttributeError:
        try:
            segments.append(cb.__name__)
            if inspect.ismethod(cb):
                try:
                    segments.insert(0, cb.im_class.__name__)
                except AttributeError:
                    pass
        except AttributeError:
            pass
    if not segments:
        return repr(cb)
    else:
        try:
            if cb.__module__:
                segments.insert(0, cb.__module__)
        except AttributeError:
            pass
        return '.'.join(segments)
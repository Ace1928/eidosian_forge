import contextlib
def loadIO_source(buffer, **kwds):
    """load an object that was stored with dill.temp.dumpIO_source

    buffer: buffer object
    alias: string name of stored object

    >>> f = lambda x:x**2
    >>> pyfile = dill.temp.dumpIO_source(f, alias='_f')
    >>> _f = dill.temp.loadIO_source(pyfile)
    >>> _f(4)
    16
    """
    alias = kwds.pop('alias', None)
    source = getattr(buffer, 'getvalue', buffer)
    if source != buffer:
        source = source()
    source = source.decode()
    if not alias:
        tag = source.strip().splitlines()[-1].split()
        if tag[0] != '#NAME:':
            stub = source.splitlines()[0]
            raise IOError('unknown name for code: %s' % stub)
        alias = tag[-1]
    local = {}
    exec(source, local)
    _ = eval('%s' % alias, local)
    return _
import contextlib
def loadIO(buffer, **kwds):
    """load an object that was stored with dill.temp.dumpIO

    buffer: buffer object

    >>> dumpfile = dill.temp.dumpIO([1, 2, 3, 4, 5])
    >>> dill.temp.loadIO(dumpfile)
    [1, 2, 3, 4, 5]
    """
    import dill as pickle
    from io import BytesIO as StringIO
    value = getattr(buffer, 'getvalue', buffer)
    if value != buffer:
        value = value()
    return pickle.load(StringIO(value))
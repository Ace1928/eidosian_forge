import contextlib
def dumpIO(object, **kwds):
    """dill.dump of object to a buffer.
Loads with "dill.temp.loadIO".  Returns the buffer object.

    >>> dumpfile = dill.temp.dumpIO([1, 2, 3, 4, 5])
    >>> dill.temp.loadIO(dumpfile)
    [1, 2, 3, 4, 5]
    """
    import dill as pickle
    from io import BytesIO as StringIO
    file = StringIO()
    pickle.dump(object, file)
    file.flush()
    return file
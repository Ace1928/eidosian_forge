from .python3_compat import iterkeys, iteritems, Mapping  #, u
def fromJSON(cls, stream, *args, **kwargs):
    """ Deserializes JSON to Munch or any of its subclasses.
        """
    factory = lambda d: cls(*args + (d,), **kwargs)
    return munchify(json.loads(stream), factory=factory)
def hashkey(*args, **kwargs):
    """Return a cache key for the specified hashable arguments."""
    if kwargs:
        return _HashedTuple(args + sum(sorted(kwargs.items()), _kwmark))
    else:
        return _HashedTuple(args)
import sys
def coroutine(func):
    """Convert regular generator function to a coroutine."""
    if not callable(func):
        raise TypeError('types.coroutine() expects a callable')
    if func.__class__ is FunctionType and getattr(func, '__code__', None).__class__ is CodeType:
        co_flags = func.__code__.co_flags
        if co_flags & 384:
            return func
        if co_flags & 32:
            co = func.__code__
            func.__code__ = co.replace(co_flags=co.co_flags | 256)
            return func
    import functools
    import _collections_abc

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        coro = func(*args, **kwargs)
        if coro.__class__ is CoroutineType or (coro.__class__ is GeneratorType and coro.gi_code.co_flags & 256):
            return coro
        if isinstance(coro, _collections_abc.Generator) and (not isinstance(coro, _collections_abc.Coroutine)):
            return _GeneratorWrapper(coro)
        return coro
    return wrapped
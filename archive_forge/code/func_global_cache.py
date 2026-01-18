import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def global_cache(srctype, ffi, funcname, *args, **kwds):
    key = kwds.pop('key', (funcname, args))
    assert not kwds
    try:
        return ffi._typecache[key]
    except KeyError:
        pass
    try:
        res = getattr(ffi._backend, funcname)(*args)
    except NotImplementedError as e:
        raise NotImplementedError('%s: %r: %s' % (funcname, srctype, e))
    cache = ffi._typecache
    with global_lock:
        res1 = cache.get(key)
        if res1 is None:
            cache[key] = res
            return res
        else:
            return res1
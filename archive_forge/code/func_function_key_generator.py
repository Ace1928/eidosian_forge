from hashlib import sha1
from ..util import compat
from ..util import langhelpers
def function_key_generator(namespace, fn, to_str=str):
    """Return a function that generates a string
    key, based on a given function as well as
    arguments to the returned function itself.

    This is used by :meth:`.CacheRegion.cache_on_arguments`
    to generate a cache key from a decorated function.

    An alternate function may be used by specifying
    the :paramref:`.CacheRegion.function_key_generator` argument
    for :class:`.CacheRegion`.

    .. seealso::

        :func:`.kwarg_function_key_generator` - similar function that also
        takes keyword arguments into account

    """
    if namespace is None:
        namespace = '%s:%s' % (fn.__module__, fn.__name__)
    else:
        namespace = '%s:%s|%s' % (fn.__module__, fn.__name__, namespace)
    args = compat.inspect_getargspec(fn)
    has_self = args[0] and args[0][0] in ('self', 'cls')

    def generate_key(*args, **kw):
        if kw:
            raise ValueError("dogpile.cache's default key creation function does not accept keyword arguments.")
        if has_self:
            args = args[1:]
        return namespace + '|' + ' '.join(map(to_str, args))
    return generate_key
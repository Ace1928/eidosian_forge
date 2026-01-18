from hashlib import sha1
from ..util import compat
from ..util import langhelpers
def function_multi_key_generator(namespace, fn, to_str=str):
    if namespace is None:
        namespace = '%s:%s' % (fn.__module__, fn.__name__)
    else:
        namespace = '%s:%s|%s' % (fn.__module__, fn.__name__, namespace)
    args = compat.inspect_getargspec(fn)
    has_self = args[0] and args[0][0] in ('self', 'cls')

    def generate_keys(*args, **kw):
        if kw:
            raise ValueError("dogpile.cache's default key creation function does not accept keyword arguments.")
        if has_self:
            args = args[1:]
        return [namespace + '|' + key for key in map(to_str, args)]
    return generate_keys
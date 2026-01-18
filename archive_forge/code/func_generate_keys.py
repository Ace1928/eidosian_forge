from hashlib import sha1
from ..util import compat
from ..util import langhelpers
def generate_keys(*args, **kw):
    if kw:
        raise ValueError("dogpile.cache's default key creation function does not accept keyword arguments.")
    if has_self:
        args = args[1:]
    return [namespace + '|' + key for key in map(to_str, args)]
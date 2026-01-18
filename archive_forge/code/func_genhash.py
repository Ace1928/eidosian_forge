import sys
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_native_str, str_consteq
from passlib.utils.compat import unicode, u, unicode_or_bytes_types
import passlib.utils.handlers as uh
@uh.deprecated_method(deprecated='1.7', removed='2.0')
@classmethod
def genhash(cls, secret, config, encoding=None):
    if not cls.identify(config):
        raise uh.exc.InvalidHashError(cls)
    return cls.hash(secret, encoding=encoding)
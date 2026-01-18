from __future__ import with_statement, absolute_import
import logging; log = logging.getLogger(__name__)
from passlib.crypto import scrypt as _scrypt
from passlib.utils import h64, to_bytes
from passlib.utils.binary import h64, b64s_decode, b64s_encode
from passlib.utils.compat import u, bascii_to_str, suppress_cause
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
@classmethod
def has_backend(cls, name='any'):
    try:
        cls.set_backend(name, dryrun=True)
        return True
    except uh.exc.MissingBackendError:
        return False
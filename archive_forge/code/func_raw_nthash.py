from binascii import hexlify
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_unicode, right_pad_string
from passlib.utils.compat import unicode
from passlib.crypto.digest import lookup_hash
import passlib.utils.handlers as uh
@classmethod
def raw_nthash(cls, secret, hex=False):
    warn('nthash.raw_nthash() is deprecated, and will be removed in Passlib 1.8, please use nthash.raw() instead', DeprecationWarning)
    ret = nthash.raw(secret)
    return hexlify(ret).decode('ascii') if hex else ret
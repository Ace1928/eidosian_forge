import logging; log = logging.getLogger(__name__)
from passlib.utils import consteq, saslprep, to_native_str, splitcomma
from passlib.utils.binary import ab64_decode, ab64_encode
from passlib.utils.compat import bascii_to_str, iteritems, u, native_string_types
from passlib.crypto.digest import pbkdf2_hmac, norm_hash_name
import passlib.utils.handlers as uh
@classmethod
def extract_digest_algs(cls, hash, format='iana'):
    """Return names of all algorithms stored in a given hash.

        :type hash: str
        :arg hash:
            The :class:`!scram` hash to parse

        :type format: str
        :param format:
            This changes the naming convention used by the
            returned algorithm names. By default the names
            are IANA-compatible; possible values are ``"iana"`` or ``"hashlib"``.

        :returns:
            Returns a list of digest algorithms; e.g. ``["sha-1"]``
        """
    algs = cls.from_string(hash).algs
    if format == 'iana':
        return algs
    else:
        return [norm_hash_name(alg, format) for alg in algs]
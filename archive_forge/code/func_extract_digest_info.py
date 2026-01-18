import logging; log = logging.getLogger(__name__)
from passlib.utils import consteq, saslprep, to_native_str, splitcomma
from passlib.utils.binary import ab64_decode, ab64_encode
from passlib.utils.compat import bascii_to_str, iteritems, u, native_string_types
from passlib.crypto.digest import pbkdf2_hmac, norm_hash_name
import passlib.utils.handlers as uh
@classmethod
def extract_digest_info(cls, hash, alg):
    """return (salt, rounds, digest) for specific hash algorithm.

        :type hash: str
        :arg hash:
            :class:`!scram` hash stored for desired user

        :type alg: str
        :arg alg:
            Name of digest algorithm (e.g. ``"sha-1"``) requested by client.

            This value is run through :func:`~passlib.crypto.digest.norm_hash_name`,
            so it is case-insensitive, and can be the raw SCRAM
            mechanism name (e.g. ``"SCRAM-SHA-1"``), the IANA name,
            or the hashlib name.

        :raises KeyError:
            If the hash does not contain an entry for the requested digest
            algorithm.

        :returns:
            A tuple containing ``(salt, rounds, digest)``,
            where *digest* matches the raw bytes returned by
            SCRAM's :func:`Hi` function for the stored password,
            the provided *salt*, and the iteration count (*rounds*).
            *salt* and *digest* are both raw (unencoded) bytes.
        """
    alg = norm_hash_name(alg, 'iana')
    self = cls.from_string(hash)
    chkmap = self.checksum
    if not chkmap:
        raise ValueError('scram hash contains no digests')
    return (self.salt, self.rounds, chkmap[alg])
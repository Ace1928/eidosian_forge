from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Util.py3compat import bchr, is_bytes
from Cryptodome.PublicKey.ECC import (EccKey,
def import_public_key(encoded):
    """Create a new Ed25519 or Ed448 public key object,
    starting from the key encoded as raw ``bytes``,
    in the format described in RFC8032.

    Args:
      encoded (bytes):
        The EdDSA public key to import.
        It must be 32 bytes for Ed25519, and 57 bytes for Ed448.

    Returns:
      :class:`Cryptodome.PublicKey.EccKey` : a new ECC key object.

    Raises:
      ValueError: when the given key cannot be parsed.
    """
    if len(encoded) == 32:
        x, y = _import_ed25519_public_key(encoded)
        curve_name = 'Ed25519'
    elif len(encoded) == 57:
        x, y = _import_ed448_public_key(encoded)
        curve_name = 'Ed448'
    else:
        raise ValueError('Not an EdDSA key (%d bytes)' % len(encoded))
    return construct(curve=curve_name, point_x=x, point_y=y)
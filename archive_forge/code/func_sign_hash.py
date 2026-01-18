import hashlib
import os
from rsa._compat import range
from rsa import common, transform, core
def sign_hash(hash_value, priv_key, hash_method):
    """Signs a precomputed hash with the private key.

    Hashes the message, then signs the hash with the given key. This is known
    as a "detached signature", because the message itself isn't altered.

    :param hash_value: A precomputed hash to sign (ignores message). Should be set to
        None if needing to hash and sign message.
    :param priv_key: the :py:class:`rsa.PrivateKey` to sign with
    :param hash_method: the hash method used on the message. Use 'MD5', 'SHA-1',
        'SHA-224', SHA-256', 'SHA-384' or 'SHA-512'.
    :return: a message signature block.
    :raise OverflowError: if the private key is too small to contain the
        requested hash.

    """
    if hash_method not in HASH_ASN1:
        raise ValueError('Invalid hash method: %s' % hash_method)
    asn1code = HASH_ASN1[hash_method]
    cleartext = asn1code + hash_value
    keylength = common.byte_size(priv_key.n)
    padded = _pad_for_signing(cleartext, keylength)
    payload = transform.bytes2int(padded)
    encrypted = priv_key.blinded_encrypt(payload)
    block = transform.int2bytes(encrypted, keylength)
    return block
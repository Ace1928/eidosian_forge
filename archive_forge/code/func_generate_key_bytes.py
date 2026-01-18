import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
def generate_key_bytes(hash_alg, salt, key, nbytes):
    """
    Given a password, passphrase, or other human-source key, scramble it
    through a secure hash into some keyworthy bytes.  This specific algorithm
    is used for encrypting/decrypting private key files.

    :param function hash_alg: A function which creates a new hash object, such
        as ``hashlib.sha256``.
    :param salt: data to salt the hash with.
    :type bytes salt: Hash salt bytes.
    :param str key: human-entered password or passphrase.
    :param int nbytes: number of bytes to generate.
    :return: Key data, as `bytes`.
    """
    keydata = bytes()
    digest = bytes()
    if len(salt) > 8:
        salt = salt[:8]
    while nbytes > 0:
        hash_obj = hash_alg()
        if len(digest) > 0:
            hash_obj.update(digest)
        hash_obj.update(b(key))
        hash_obj.update(salt)
        digest = hash_obj.digest()
        size = min(nbytes, len(digest))
        keydata += digest[:size]
        nbytes -= size
    return keydata
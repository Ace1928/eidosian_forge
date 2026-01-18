from base64 import encodebytes, decodebytes
import binascii
import os
import re
from collections.abc import MutableMapping
from hashlib import sha1
from hmac import HMAC
from paramiko.pkey import PKey, UnknownKeyType
from paramiko.util import get_logger, constant_time_bytes_eq, b, u
from paramiko.ssh_exception import SSHException
@staticmethod
def hash_host(hostname, salt=None):
    """
        Return a "hashed" form of the hostname, as used by OpenSSH when storing
        hashed hostnames in the known_hosts file.

        :param str hostname: the hostname to hash
        :param str salt: optional salt to use when hashing
            (must be 20 bytes long)
        :return: the hashed hostname as a `str`
        """
    if salt is None:
        salt = os.urandom(sha1().digest_size)
    else:
        if salt.startswith('|1|'):
            salt = salt.split('|')[2]
        salt = decodebytes(b(salt))
    assert len(salt) == sha1().digest_size
    hmac = HMAC(salt, b(hostname), sha1).digest()
    hostkey = '|1|{}|{}'.format(u(encodebytes(salt)), u(encodebytes(hmac)))
    return hostkey.replace('\n', '')
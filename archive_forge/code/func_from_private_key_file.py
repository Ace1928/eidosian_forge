import base64
from base64 import encodebytes, decodebytes
from binascii import unhexlify
import os
from pathlib import Path
from hashlib import md5, sha256
import re
import struct
import bcrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import algorithms, modes, Cipher
from cryptography.hazmat.primitives import asymmetric
from paramiko import util
from paramiko.util import u, b
from paramiko.common import o600
from paramiko.ssh_exception import SSHException, PasswordRequiredException
from paramiko.message import Message
@classmethod
def from_private_key_file(cls, filename, password=None):
    """
        Create a key object by reading a private key file.  If the private
        key is encrypted and ``password`` is not ``None``, the given password
        will be used to decrypt the key (otherwise `.PasswordRequiredException`
        is thrown).  Through the magic of Python, this factory method will
        exist in all subclasses of PKey (such as `.RSAKey` or `.DSSKey`), but
        is useless on the abstract PKey class.

        :param str filename: name of the file to read
        :param str password:
            an optional password to use to decrypt the key file, if it's
            encrypted
        :return: a new `.PKey` based on the given private key

        :raises: ``IOError`` -- if there was an error reading the file
        :raises: `.PasswordRequiredException` -- if the private key file is
            encrypted, and ``password`` is ``None``
        :raises: `.SSHException` -- if the key file is invalid
        """
    key = cls(filename=filename, password=password)
    return key
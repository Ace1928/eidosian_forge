import re
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, HMAC, SHA256, MD5, SHA224, SHA384, SHA512
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Protocol.KDF import (PBKDF1, PBKDF2, _S2V, HKDF, scrypt,
from Cryptodome.Protocol.KDF import _bcrypt_decode
def test_empty_password(self):
    tvs = [(b'', 4, b'zVHmKQtGGQob.b/Nc7l9NO', b'$2a$04$zVHmKQtGGQob.b/Nc7l9NO8UlrYcW05FiuCj/SxsFO/ZtiN9.mNzy'), (b'', 5, b'zVHmKQtGGQob.b/Nc7l9NO', b'$2a$05$zVHmKQtGGQob.b/Nc7l9NOWES.1hkVBgy5IWImh9DOjKNU8atY4Iy'), (b'', 6, b'zVHmKQtGGQob.b/Nc7l9NO', b'$2a$06$zVHmKQtGGQob.b/Nc7l9NOjOl7l4oz3WSh5fJ6414Uw8IXRAUoiaO'), (b'', 7, b'zVHmKQtGGQob.b/Nc7l9NO', b'$2a$07$zVHmKQtGGQob.b/Nc7l9NOBsj1dQpBA1HYNGpIETIByoNX9jc.hOi'), (b'', 8, b'zVHmKQtGGQob.b/Nc7l9NO', b'$2a$08$zVHmKQtGGQob.b/Nc7l9NOiLTUh/9MDpX86/DLyEzyiFjqjBFePgO')]
    for idx, (password, cost, salt64, result) in enumerate(tvs):
        x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
        self.assertEqual(x, result)
        bcrypt_check(password, result)
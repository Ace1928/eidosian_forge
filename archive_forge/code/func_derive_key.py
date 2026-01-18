from __future__ import annotations
import collections.abc as cabc
import hashlib
import hmac
import typing as t
from .encoding import _base64_alphabet
from .encoding import base64_decode
from .encoding import base64_encode
from .encoding import want_bytes
from .exc import BadSignature
def derive_key(self, secret_key: str | bytes | None=None) -> bytes:
    """This method is called to derive the key. The default key
        derivation choices can be overridden here. Key derivation is not
        intended to be used as a security method to make a complex key
        out of a short password. Instead you should use large random
        secret keys.

        :param secret_key: A specific secret key to derive from.
            Defaults to the last item in :attr:`secret_keys`.

        .. versionchanged:: 2.0
            Added the ``secret_key`` parameter.
        """
    if secret_key is None:
        secret_key = self.secret_keys[-1]
    else:
        secret_key = want_bytes(secret_key)
    if self.key_derivation == 'concat':
        return t.cast(bytes, self.digest_method(self.salt + secret_key).digest())
    elif self.key_derivation == 'django-concat':
        return t.cast(bytes, self.digest_method(self.salt + b'signer' + secret_key).digest())
    elif self.key_derivation == 'hmac':
        mac = hmac.new(secret_key, digestmod=self.digest_method)
        mac.update(self.salt)
        return mac.digest()
    elif self.key_derivation == 'none':
        return secret_key
    else:
        raise TypeError('Unknown key derivation method')
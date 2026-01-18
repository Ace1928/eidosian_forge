from __future__ import annotations
import typing
def cryptography_has_300_evp_cipher() -> typing.List[str]:
    return ['EVP_CIPHER_fetch', 'EVP_CIPHER_free']
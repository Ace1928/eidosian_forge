from __future__ import annotations
import typing
def cryptography_has_evp_aead() -> typing.List[str]:
    return ['EVP_aead_chacha20_poly1305', 'EVP_AEAD_CTX_free', 'EVP_AEAD_CTX_seal', 'EVP_AEAD_CTX_open', 'EVP_AEAD_max_overhead', 'Cryptography_EVP_AEAD_CTX_new']
from __future__ import annotations
import typing
def cryptography_has_raw_key() -> typing.List[str]:
    return ['EVP_PKEY_new_raw_private_key', 'EVP_PKEY_new_raw_public_key', 'EVP_PKEY_get_raw_private_key', 'EVP_PKEY_get_raw_public_key']
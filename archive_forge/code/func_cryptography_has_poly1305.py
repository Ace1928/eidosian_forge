from __future__ import annotations
import typing
def cryptography_has_poly1305() -> typing.List[str]:
    return ['NID_poly1305', 'EVP_PKEY_POLY1305']
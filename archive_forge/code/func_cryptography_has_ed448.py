from __future__ import annotations
import typing
def cryptography_has_ed448() -> typing.List[str]:
    return ['EVP_PKEY_ED448', 'NID_ED448']
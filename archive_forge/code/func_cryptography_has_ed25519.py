from __future__ import annotations
import typing
def cryptography_has_ed25519() -> typing.List[str]:
    return ['NID_ED25519', 'EVP_PKEY_ED25519']
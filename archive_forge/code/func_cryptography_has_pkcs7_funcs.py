from __future__ import annotations
import typing
def cryptography_has_pkcs7_funcs() -> typing.List[str]:
    return ['SMIME_write_PKCS7', 'PEM_write_bio_PKCS7_stream', 'PKCS7_sign_add_signer', 'PKCS7_final', 'PKCS7_verify', 'SMIME_read_PKCS7', 'PKCS7_get0_signers']
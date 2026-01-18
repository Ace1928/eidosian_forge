from __future__ import annotations
import typing
def cryptography_has_set_cert_cb() -> typing.List[str]:
    return ['SSL_CTX_set_cert_cb', 'SSL_set_cert_cb']
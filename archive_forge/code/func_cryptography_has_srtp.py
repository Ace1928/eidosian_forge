from __future__ import annotations
import typing
def cryptography_has_srtp() -> typing.List[str]:
    return ['SSL_CTX_set_tlsext_use_srtp', 'SSL_set_tlsext_use_srtp', 'SSL_get_selected_srtp_profile']
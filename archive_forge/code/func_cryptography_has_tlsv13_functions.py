from __future__ import annotations
import typing
def cryptography_has_tlsv13_functions() -> typing.List[str]:
    return ['SSL_VERIFY_POST_HANDSHAKE', 'SSL_CTX_set_ciphersuites', 'SSL_verify_client_post_handshake', 'SSL_CTX_set_post_handshake_auth', 'SSL_set_post_handshake_auth', 'SSL_SESSION_get_max_early_data', 'SSL_write_early_data', 'SSL_read_early_data', 'SSL_CTX_set_max_early_data']
from __future__ import annotations
import typing
def cryptography_has_ssl_st() -> typing.List[str]:
    return ['SSL_ST_BEFORE', 'SSL_ST_OK', 'SSL_ST_INIT', 'SSL_ST_RENEGOTIATE']
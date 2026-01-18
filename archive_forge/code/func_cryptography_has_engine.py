from __future__ import annotations
import typing
def cryptography_has_engine() -> typing.List[str]:
    return ['ENGINE_by_id', 'ENGINE_init', 'ENGINE_finish', 'ENGINE_get_default_RAND', 'ENGINE_set_default_RAND', 'ENGINE_unregister_RAND', 'ENGINE_ctrl_cmd', 'ENGINE_free', 'ENGINE_get_name', 'ENGINE_ctrl_cmd_string', 'ENGINE_load_builtin_engines', 'ENGINE_load_private_key', 'ENGINE_load_public_key', 'SSL_CTX_set_client_cert_engine']
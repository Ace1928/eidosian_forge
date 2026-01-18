from __future__ import annotations
import typing
def cryptography_has_providers() -> typing.List[str]:
    return ['OSSL_PROVIDER_load', 'OSSL_PROVIDER_unload', 'ERR_LIB_PROV', 'PROV_R_WRONG_FINAL_BLOCK_LENGTH', 'PROV_R_BAD_DECRYPT']
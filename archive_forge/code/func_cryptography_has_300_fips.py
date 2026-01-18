from __future__ import annotations
import typing
def cryptography_has_300_fips() -> typing.List[str]:
    return ['EVP_default_properties_is_fips_enabled', 'EVP_default_properties_enable_fips']
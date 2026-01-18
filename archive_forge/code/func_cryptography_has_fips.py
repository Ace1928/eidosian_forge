from __future__ import annotations
import typing
def cryptography_has_fips() -> typing.List[str]:
    return ['FIPS_mode_set', 'FIPS_mode']
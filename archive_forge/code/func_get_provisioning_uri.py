from __future__ import annotations
import typing
from cryptography.hazmat.primitives import constant_time
from cryptography.hazmat.primitives.twofactor import InvalidToken
from cryptography.hazmat.primitives.twofactor.hotp import (
def get_provisioning_uri(self, account_name: str, issuer: typing.Optional[str]) -> str:
    return _generate_uri(self._hotp, 'totp', account_name, issuer, [('period', int(self._time_step))])
from __future__ import annotations
import os
from typing import ClassVar
from ._typing import Literal
from ._utils import Parameters, _check_types, extract_parameters
from .exceptions import InvalidHashError
from .low_level import Type, hash_secret, verify_secret
from .profiles import RFC_9106_LOW_MEMORY
def check_needs_rehash(self, hash: str) -> bool:
    """
        Check whether *hash* was created using the instance's parameters.

        Whenever your Argon2 parameters -- or *argon2-cffi*'s defaults! --
        change, you should rehash your passwords at the next opportunity.  The
        common approach is to do that whenever a user logs in, since that
        should be the only time when you have access to the cleartext
        password.

        Therefore it's best practice to check -- and if necessary rehash --
        passwords after each successful authentication.

        :rtype: bool

        .. versionadded:: 18.2.0
        """
    return self._parameters != extract_parameters(hash)
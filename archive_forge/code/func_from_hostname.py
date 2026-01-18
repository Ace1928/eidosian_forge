import os
from enum import Enum
from typing import Any
@classmethod
def from_hostname(cls, hostname: str) -> 'AppEnv':
    """
        Get the app environment from the hostname
        """
    hostname = hostname.lower()
    if 'dev' in hostname:
        return cls.DEVELOPMENT
    if 'staging' in hostname:
        return cls.STAGING
    return cls.LOCAL if 'local' in hostname else cls.PRODUCTION
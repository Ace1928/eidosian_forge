from __future__ import annotations
from typing import Optional
from pymongo.errors import ConfigurationError
def get_ssl_context(*dummy):
    """No ssl module, raise ConfigurationError."""
    raise ConfigurationError('The ssl module is not available.')
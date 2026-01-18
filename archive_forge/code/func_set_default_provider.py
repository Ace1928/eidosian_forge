import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def set_default_provider(self, key, default_provider):
    provider = self._override_providers.get(key)
    if isinstance(provider, ChainProvider):
        provider.set_default_provider(default_provider)
        return
    elif isinstance(provider, BaseProvider):
        default_provider = ChainProvider(providers=[provider, default_provider])
    self._override_providers[key] = default_provider
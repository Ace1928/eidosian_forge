import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def get_default_config_values(self, mode):
    if mode not in self._resolved_default_configurations:
        defaults = self._resolve_default_values_by_mode(mode)
        self._resolved_default_configurations[mode] = defaults
    return self._resolved_default_configurations[mode]
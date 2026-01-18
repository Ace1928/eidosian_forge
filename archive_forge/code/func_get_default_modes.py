import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def get_default_modes(self):
    default_modes = ['legacy', 'auto']
    default_modes.extend(self._modes.keys())
    return default_modes
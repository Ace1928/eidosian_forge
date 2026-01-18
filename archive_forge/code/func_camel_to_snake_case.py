from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def camel_to_snake_case(self, config):
    """
        Convert camel case keys to snake case keys in the config.

        Parameters:
            config (list) - Playbook details provided by the user.

        Returns:
            new_config (list) - Updated config after eliminating the camel cases.
        """
    if isinstance(config, dict):
        new_config = {}
        for key, value in config.items():
            new_key = re.sub('([a-z0-9])([A-Z])', '\\1_\\2', key).lower()
            if new_key != key:
                self.log('{0} will be deprecated soon. Please use {1}.'.format(key, new_key), 'DEBUG')
            new_value = self.camel_to_snake_case(value)
            new_config[new_key] = new_value
    elif isinstance(config, list):
        return [self.camel_to_snake_case(item) for item in config]
    else:
        return config
    return new_config
from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def check_string_dictionary(self, task_details_data):
    """
        Check whether the input is string dictionary or string.

        Parameters:
            task_details_data (string) - Input either string dictionary or string.

        Returns:
            value (dict) - If the input is string dictionary, else returns None.
        """
    try:
        value = json.loads(task_details_data)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    return None
from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib  # noqa: F401, pylint: disable=unused-import
from ansible.module_utils.six.moves import configparser
from os.path import expanduser
from uuid import UUID
@staticmethod
def argument_spec_with_wait(**additional_argument_spec):
    """
        Build an argument specification for a Dimension Data module that includes "wait for completion" arguments.
        :param additional_argument_spec: An optional dictionary representing the specification for additional module arguments (if any).
        :return: A dict containing the argument specification.
        """
    spec = DimensionDataModule.argument_spec(wait=dict(type='bool', required=False, default=False), wait_time=dict(type='int', required=False, default=600), wait_poll_interval=dict(type='int', required=False, default=2))
    if additional_argument_spec:
        spec.update(additional_argument_spec)
    return spec
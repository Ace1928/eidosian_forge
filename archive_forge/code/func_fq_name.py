from __future__ import absolute_import, division, print_function
import copy
import os
import re
import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible.module_utils.parsing.convert_bool import (
from collections import defaultdict
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from .constants import (
def fq_name(partition, value, sub_path=''):
    """Returns a 'Fully Qualified' name

    A BIG-IP expects most names of resources to be in a fully-qualified
    form. This means that both the simple name, and the partition need
    to be combined.

    The Ansible modules, however, can accept (as names for several
    resources) their name in the FQ format. This becomes an issue when
    the FQ name and the partition are both specified as separate values.

    Consider the following examples.

        # Name not FQ
        name: foo
        partition: Common

        # Name FQ
        name: /Common/foo
        partition: Common

    This method will rectify the above situation and will, in both cases,
    return the following for name.

        /Common/foo

    Args:
        partition (string): The partition that you would want attached to
            the name if the name has no partition.
        value (string): The name that you want to attach a partition to.
            This value will be returned unchanged if it has a partition
            attached to it already.
        sub_path (string): The sub path element. If defined the sub_path
            will be inserted between partition and value.
            This will also work on FQ names.
    Returns:
        string: The fully qualified name, given the input parameters.
    """
    if value is not None and sub_path == '':
        try:
            int(value)
            return '/{0}/{1}'.format(partition, value)
        except (ValueError, TypeError):
            if not value.startswith('/'):
                return '/{0}/{1}'.format(partition, value)
    if value is not None and sub_path != '':
        try:
            int(value)
            return '/{0}/{1}/{2}'.format(partition, sub_path, value)
        except (ValueError, TypeError):
            if value.startswith('/'):
                dummy, partition, name = value.split('/')
                return '/{0}/{1}/{2}'.format(partition, sub_path, name)
            if not value.startswith('/'):
                return '/{0}/{1}/{2}'.format(partition, sub_path, value)
    return value
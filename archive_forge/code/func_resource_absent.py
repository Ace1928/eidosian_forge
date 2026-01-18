from __future__ import (absolute_import, division, print_function)
import abc
import collections
import json
import os  # noqa: F401, pylint: disable=unused-import
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._collections_compat import Mapping
def resource_absent(self, resource, method='delete'):
    """
        Generic implementation of the absent state for the OneView resources.

        It checks if the resource needs to be removed.

        :arg dict resource: Resource to delete.
        :arg str method: Function of the OneView client that will be called for resource deletion.
            Usually delete or remove.
        :return: A dictionary with the expected arguments for the AnsibleModule.exit_json
        """
    if resource:
        getattr(self.resource_client, method)(resource)
        return {'changed': True, 'msg': self.MSG_DELETED}
    else:
        return {'changed': False, 'msg': self.MSG_ALREADY_ABSENT}
from __future__ import (absolute_import, division, print_function)
import abc
import os
import platform
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE, BOOLEANS_FALSE
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
def get_container_by_id(self, container_id):
    try:
        self.log('Inspecting container Id %s' % container_id)
        result = self.inspect_container(container=container_id)
        self.log('Completed container inspection')
        return result
    except NotFound as dummy:
        return None
    except Exception as exc:
        self.fail('Error inspecting container: %s' % exc)
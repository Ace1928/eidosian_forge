from __future__ import (absolute_import, division, print_function)
import abc
import os
import re
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE, BOOLEANS_FALSE
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api import auth
from ansible_collections.community.docker.plugins.module_utils._api.api.client import APIClient as Client
from ansible_collections.community.docker.plugins.module_utils._api.errors import (
from ansible_collections.community.docker.plugins.module_utils._api.tls import TLSConfig
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
def depr(*args, **kwargs):
    self.deprecate(*args, **kwargs)
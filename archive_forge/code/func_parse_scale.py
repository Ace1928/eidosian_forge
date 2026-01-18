from __future__ import absolute_import, division, print_function
import os
import re
import sys
import tempfile
import traceback
from contextlib import contextmanager
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
def parse_scale(self, service_name):
    try:
        return int(self.scale[service_name])
    except ValueError:
        self.client.fail('Error scaling %s - expected int, got %s', service_name, to_native(type(self.scale[service_name])))
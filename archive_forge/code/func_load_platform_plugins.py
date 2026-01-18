from __future__ import absolute_import, division, print_function
from io import BytesIO
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import cPickle
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.playbook.play_context import PlayContext
from ansible.plugins.connection import ensure_connect
from ansible.plugins.loader import httpapi_loader
from ansible.release import __version__ as ANSIBLE_CORE_VERSION
from ansible_collections.ansible.netcommon.plugins.plugin_utils.connection_base import (
from ansible_collections.ansible.netcommon.plugins.plugin_utils.version import Version
def load_platform_plugins(self, platform_type=None):
    platform_type = platform_type or self.get_option('platform_type')
    if platform_type:
        self.httpapi = httpapi_loader.get(platform_type, self)
        if self.httpapi:
            self._sub_plugin = {'type': 'httpapi', 'name': self.httpapi._load_name, 'obj': self.httpapi}
            self.queue_message('vvvv', 'loaded API plugin %s from path %s for platform type %s' % (self.httpapi._load_name, self.httpapi._original_path, platform_type))
        else:
            raise AnsibleConnectionFailure('unable to load API plugin for platform type %s' % platform_type)
    else:
        raise AnsibleConnectionFailure('Unable to automatically determine host platform type. Please manually configure platform_type value for this host')
    self.queue_message('log', 'platform_type is set to %s' % platform_type)
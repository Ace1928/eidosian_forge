from __future__ import absolute_import, division, print_function
import hashlib
import os
import posixpath
import shutil
import io
import tempfile
import traceback
import re
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.ansible_release import __version__ as ansible_version
from re import match
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def find_latest_version_available(self, artifact):
    if self.latest_version_found:
        return self.latest_version_found
    path = '/%s/%s' % (artifact.path(False), self.metadata_file_name)
    content = self._getContent(self.base + path, 'Failed to retrieve the maven metadata file: ' + path)
    xml = etree.fromstring(content)
    v = xml.xpath('/metadata/versioning/versions/version[last()]/text()')
    if v:
        self.latest_version_found = v[0]
        return v[0]
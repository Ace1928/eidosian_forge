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
def find_version_by_spec(self, artifact):
    path = '/%s/%s' % (artifact.path(False), self.metadata_file_name)
    content = self._getContent(self.base + path, 'Failed to retrieve the maven metadata file: ' + path)
    xml = etree.fromstring(content)
    original_versions = xml.xpath('/metadata/versioning/versions/version/text()')
    versions = []
    for version in original_versions:
        try:
            versions.append(Version.coerce(version))
        except ValueError:
            pass
    parse_versions_syntax = {'^\\(,(?P<upper_bound>[0-9.]*)]$': '<={upper_bound}', '^(?P<version>[0-9.]*)$': '~={version}', '^\\[(?P<version>[0-9.]*)\\]$': '=={version}', '^\\[(?P<lower_bound>[0-9.]*),\\s*(?P<upper_bound>[0-9.]*)\\]$': '>={lower_bound},<={upper_bound}', '^\\[(?P<lower_bound>[0-9.]*),\\s*(?P<upper_bound>[0-9.]+)\\)$': '>={lower_bound},<{upper_bound}', '^\\[(?P<lower_bound>[0-9.]*),\\)$': '>={lower_bound}'}
    for regex, spec_format in parse_versions_syntax.items():
        regex_result = match(regex, artifact.version_by_spec)
        if regex_result:
            spec = Spec(spec_format.format(**regex_result.groupdict()))
            selected_version = spec.select(versions)
            if not selected_version:
                raise ValueError('No version found with this spec version: {0}'.format(artifact.version_by_spec))
            if str(selected_version) not in original_versions:
                selected_version.patch = None
            return str(selected_version)
    raise ValueError('The spec version {0} is not supported! '.format(artifact.version_by_spec))
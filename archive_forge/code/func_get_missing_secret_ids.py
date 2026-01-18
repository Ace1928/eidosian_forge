from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def get_missing_secret_ids(self):
    """
        Resolve missing secret ids by looking them up by name
        """
    secret_names = [secret['secret_name'] for secret in self.client.module.params.get('secrets') or [] if secret['secret_id'] is None]
    if not secret_names:
        return {}
    secrets = self.client.secrets(filters={'name': secret_names})
    secrets = dict(((secret['Spec']['Name'], secret['ID']) for secret in secrets if secret['Spec']['Name'] in secret_names))
    for secret_name in secret_names:
        if secret_name not in secrets:
            self.client.fail('Could not find a secret named "%s"' % secret_name)
    return secrets
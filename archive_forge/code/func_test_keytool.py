from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def test_keytool(module, executable):
    """ Test if keytool is actually executable or not """
    module.run_command([executable], check_rc=True)
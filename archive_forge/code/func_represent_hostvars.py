from __future__ import (absolute_import, division, print_function)
import yaml
from ansible.module_utils.six import text_type, binary_type
from ansible.module_utils.common.yaml import SafeDumper
from ansible.parsing.yaml.objects import AnsibleUnicode, AnsibleSequence, AnsibleMapping, AnsibleVaultEncryptedUnicode
from ansible.utils.unsafe_proxy import AnsibleUnsafeText, AnsibleUnsafeBytes, NativeJinjaUnsafeText, NativeJinjaText, _is_unsafe
from ansible.template import AnsibleUndefined
from ansible.vars.hostvars import HostVars, HostVarsVars
from ansible.vars.manager import VarsWithSources
def represent_hostvars(self, data):
    return self.represent_dict(dict(data))
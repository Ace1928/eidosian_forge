from __future__ import absolute_import, division, print_function
import os
import hashlib
from time import sleep
from threading import Thread
from ansible.module_utils.urls import open_url
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def set_progress_percent(self, progress_percent):
    self.progressPercent = progress_percent
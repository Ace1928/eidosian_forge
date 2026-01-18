from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def build_job(description):
    return otypes.Job(description=description, status=otypes.JobStatus.STARTED, external=True, auto_cleared=True)
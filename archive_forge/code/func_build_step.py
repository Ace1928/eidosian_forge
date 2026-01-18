from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def build_step(description, job_id):
    return otypes.Step(description=description, type=otypes.StepEnum.UNKNOWN, job=otypes.Job(id=job_id), status=otypes.StepStatus.STARTED, external=True)
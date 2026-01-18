from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def attach_steps(module, job_id, jobs_service):
    changed = False
    steps_service = jobs_service.job_service(job_id).steps_service()
    if module.params.get('steps'):
        for step in module.params.get('steps'):
            step_entity = get_entity(steps_service, step.get('description'))
            step_state = step.get('state', 'present')
            if step_state in ['present', 'started']:
                if step_entity is None:
                    steps_service.add(build_step(step.get('description'), job_id))
                    changed = True
            if step_entity is not None and step_entity.status not in [otypes.StepStatus.FINISHED, otypes.StepStatus.FAILED]:
                if step_state in ['absent', 'finished']:
                    steps_service.step_service(step_entity.id).end(succeeded=True)
                    changed = True
                elif step_state == 'failed':
                    steps_service.step_service(step_entity.id).end(succeeded=False)
                    changed = True
    return changed
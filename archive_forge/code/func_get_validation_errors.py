from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_validation_errors(self, validation_errors):
    """
        Store template parameters from the playbook for template processing in DNAC.

        Parameters:
            validation_errors (dict) - Playbook details containing validation errors information.

        Returns:
            validationErrors (dict) - Organized validation errors parameters.
        """
    if validation_errors is None:
        return None
    validationErrors = {}
    rollback_template_errors = validation_errors.get('rollback_template_errors')
    if rollback_template_errors is not None:
        validationErrors.update({'rollbackTemplateErrors': rollback_template_errors})
    template_errors = validation_errors.get('template_errors')
    if template_errors is not None:
        validationErrors.update({'templateErrors': template_errors})
    template_id = validation_errors.get('template_id')
    if template_id is not None:
        validationErrors.update({'templateId': template_id})
    template_version = validation_errors.get('template_version')
    if template_version is not None:
        validationErrors.update({'templateVersion': template_version})
    return validationErrors
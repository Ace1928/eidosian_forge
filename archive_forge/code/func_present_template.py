from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_template(self):
    template = self.get_template()
    if template:
        template = self.update_template(template)
    elif self.module.params.get('url'):
        template = self.register_template()
    elif self.module.params.get('vm'):
        template = self.create_template()
    else:
        self.fail_json(msg='one of the following is required on state=present: url, vm')
    return template
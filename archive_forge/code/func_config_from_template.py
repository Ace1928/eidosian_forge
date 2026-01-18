from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def config_from_template(module):
    """ Load the Jinja template and apply user provided parameters in necessary
        places. Fail if template is not found. Fail if rendered template does
        not reference the correct port. Fail if the template requires a VLAN
        but the user did not provide one with the port_vlan parameter.

    :param module: Ansible module with parameters and client connection.
    :return: String of Jinja template rendered with parameters or exit with
             failure.
    """
    template_loader = jinja2.FileSystemLoader('./templates')
    env = jinja2.Environment(loader=template_loader, undefined=jinja2.DebugUndefined)
    template = env.get_template(module.params['template'])
    if not template:
        module.fail_json(msg=str('Could not find template - %s' % module.params['template']))
    data = {'switch_port': module.params['switch_port'], 'server_name': module.params['server_name']}
    temp_source = env.loader.get_source(env, module.params['template'])[0]
    parsed_content = env.parse(temp_source)
    temp_vars = list(meta.find_undeclared_variables(parsed_content))
    if 'port_vlan' in temp_vars:
        if module.params['port_vlan']:
            data['port_vlan'] = module.params['port_vlan']
        else:
            module.fail_json(msg=str('Template %s requires a vlan. Please re-run with vlan number provided.' % module.params['template']))
    template = template.render(data)
    if not valid_template(module.params['switch_port'], template):
        module.fail_json(msg=str('Template content does not configure proper interface - %s' % template))
    return template
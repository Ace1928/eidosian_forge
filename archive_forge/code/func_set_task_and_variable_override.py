from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.playbook.attribute import FieldAttribute
from ansible.playbook.base import Base
from ansible.utils.display import Display
def set_task_and_variable_override(self, task, variables, templar):
    """
        Sets attributes from the task if they are set, which will override
        those from the play.

        :arg task: the task object with the parameters that were set on it
        :arg variables: variables from inventory
        :arg templar: templar instance if templating variables is needed
        """
    new_info = self.copy()
    for attr in TASK_ATTRIBUTE_OVERRIDES:
        if (attr_val := getattr(task, attr, None)) is not None:
            setattr(new_info, attr, attr_val)
    if task.delegate_to is not None:
        delegated_host_name = templar.template(task.delegate_to)
        delegated_vars = variables.get('ansible_delegated_vars', dict()).get(delegated_host_name, dict())
        delegated_transport = C.DEFAULT_TRANSPORT
        for transport_var in C.MAGIC_VARIABLE_MAPPING.get('connection'):
            if transport_var in delegated_vars:
                delegated_transport = delegated_vars[transport_var]
                break
        for address_var in ('ansible_%s_host' % delegated_transport,) + C.MAGIC_VARIABLE_MAPPING.get('remote_addr'):
            if address_var in delegated_vars:
                break
        else:
            display.debug('no remote address found for delegated host %s\nusing its name, so success depends on DNS resolution' % delegated_host_name)
            delegated_vars['ansible_host'] = delegated_host_name
        for port_var in ('ansible_%s_port' % delegated_transport,) + C.MAGIC_VARIABLE_MAPPING.get('port'):
            if port_var in delegated_vars:
                break
        else:
            if delegated_transport == 'winrm':
                delegated_vars['ansible_port'] = 5986
            else:
                delegated_vars['ansible_port'] = C.DEFAULT_REMOTE_PORT
        for user_var in ('ansible_%s_user' % delegated_transport,) + C.MAGIC_VARIABLE_MAPPING.get('remote_user'):
            if user_var in delegated_vars and delegated_vars[user_var]:
                break
        else:
            delegated_vars['ansible_user'] = task.remote_user or self.remote_user
    else:
        delegated_vars = dict()
        for exe_var in C.MAGIC_VARIABLE_MAPPING.get('executable'):
            if exe_var in variables:
                setattr(new_info, 'executable', variables.get(exe_var))
    attrs_considered = []
    for attr, variable_names in C.MAGIC_VARIABLE_MAPPING.items():
        for variable_name in variable_names:
            if attr in attrs_considered:
                continue
            if task.delegate_to is not None:
                if isinstance(delegated_vars, dict) and variable_name in delegated_vars:
                    setattr(new_info, attr, delegated_vars[variable_name])
                    attrs_considered.append(attr)
            elif variable_name in variables:
                setattr(new_info, attr, variables[variable_name])
                attrs_considered.append(attr)
    for become_pass_name in C.MAGIC_VARIABLE_MAPPING.get('become_pass'):
        if become_pass_name in variables:
            break
    if new_info.port is None and C.DEFAULT_REMOTE_PORT is not None:
        new_info.port = int(C.DEFAULT_REMOTE_PORT)
    if len(delegated_vars) > 0:
        for connection_type in C.MAGIC_VARIABLE_MAPPING.get('connection'):
            if connection_type in delegated_vars:
                break
        else:
            remote_addr_local = new_info.remote_addr in C.LOCALHOST
            inv_hostname_local = delegated_vars.get('inventory_hostname') in C.LOCALHOST
            if remote_addr_local and inv_hostname_local:
                setattr(new_info, 'connection', 'local')
            elif getattr(new_info, 'connection', None) == 'local' and (not remote_addr_local or not inv_hostname_local):
                setattr(new_info, 'connection', C.DEFAULT_TRANSPORT)
    if new_info.connection == 'local':
        if not new_info.connection_user:
            new_info.connection_user = new_info.remote_user
    if new_info.remote_addr == 'inventory_hostname':
        new_info.remote_addr = variables.get('inventory_hostname')
        display.warning('The "%s" connection plugin has an improperly configured remote target value, forcing "inventory_hostname" templated value instead of the string' % new_info.connection)
    if task.check_mode is not None:
        new_info.check_mode = task.check_mode
    if task.diff is not None:
        new_info.diff = task.diff
    return new_info
from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak \
from ansible.module_utils.basic import AnsibleModule
def create_or_update_executions(kc, config, realm='master'):
    """
    Create or update executions for an authentication flow.
    :param kc: Keycloak API access.
    :param config: Representation of the authentication flow including it's executions.
    :param realm: Realm
    :return: tuple (changed, dict(before, after)
        WHERE
        bool changed indicates if changes have been made
        dict(str, str) shows state before and after creation/update
    """
    try:
        changed = False
        after = ''
        before = ''
        if 'authenticationExecutions' in config:
            existing_executions = kc.get_executions_representation(config, realm=realm)
            for new_exec_index, new_exec in enumerate(config['authenticationExecutions'], start=0):
                if new_exec['index'] is not None:
                    new_exec_index = new_exec['index']
                exec_found = False
                if new_exec['flowAlias'] is not None:
                    flow_alias_parent = new_exec['flowAlias']
                else:
                    flow_alias_parent = config['alias']
                exec_index = find_exec_in_executions(new_exec, existing_executions)
                if exec_index != -1:
                    exclude_key = ['flowAlias', 'subFlowType']
                    for index_key, key in enumerate(new_exec, start=0):
                        if new_exec[key] is None:
                            exclude_key.append(key)
                    if not is_struct_included(new_exec, existing_executions[exec_index], exclude_key) or exec_index != new_exec_index:
                        exec_found = True
                        if new_exec['index'] is None:
                            new_exec_index = exec_index
                        before += str(existing_executions[exec_index]) + '\n'
                    id_to_update = existing_executions[exec_index]['id']
                    existing_executions[exec_index].clear()
                elif new_exec['providerId'] is not None:
                    kc.create_execution(new_exec, flowAlias=flow_alias_parent, realm=realm)
                    exec_found = True
                    exec_index = new_exec_index
                    id_to_update = kc.get_executions_representation(config, realm=realm)[exec_index]['id']
                    after += str(new_exec) + '\n'
                elif new_exec['displayName'] is not None:
                    kc.create_subflow(new_exec['displayName'], flow_alias_parent, realm=realm, flowType=new_exec['subFlowType'])
                    exec_found = True
                    exec_index = new_exec_index
                    id_to_update = kc.get_executions_representation(config, realm=realm)[exec_index]['id']
                    after += str(new_exec) + '\n'
                if exec_found:
                    changed = True
                    if exec_index != -1:
                        updated_exec = {'id': id_to_update}
                        if new_exec['authenticationConfig'] is not None:
                            kc.add_authenticationConfig_to_execution(updated_exec['id'], new_exec['authenticationConfig'], realm=realm)
                        for key in new_exec:
                            if key not in ('flowAlias', 'authenticationConfig', 'subFlowType'):
                                updated_exec[key] = new_exec[key]
                        if new_exec['requirement'] is not None:
                            kc.update_authentication_executions(flow_alias_parent, updated_exec, realm=realm)
                        diff = exec_index - new_exec_index
                        kc.change_execution_priority(updated_exec['id'], diff, realm=realm)
                        after += str(kc.get_executions_representation(config, realm=realm)[new_exec_index]) + '\n'
        return (changed, dict(before=before, after=after))
    except Exception as e:
        kc.module.fail_json(msg='Could not create or update executions for authentication flow %s in realm %s: %s' % (config['alias'], realm, str(e)))
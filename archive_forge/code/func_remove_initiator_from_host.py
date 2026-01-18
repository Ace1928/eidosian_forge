from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def remove_initiator_from_host(self, host_details, initiators):
    """ Remove initiator from host """
    try:
        existing_initiators = self.get_host_initiators_list(host_details)
        if existing_initiators is None:
            LOG.info('No exisiting initiators in host: %s', host_details.name)
            return (False, host_details)
        if not set(initiators).issubset(set(existing_initiators)):
            LOG.info('Initiators already absent in host: %s', host_details.name)
            return (False, host_details)
        LOG.info('Removing initiators from host %s', host_details.name)
        if len(initiators) > 1:
            self.check_if_initiators_logged_in(initiators)
        for id in initiators:
            initiator_details = utils.host.UnityHostInitiatorList.get(cli=self.unity._cli, initiator_id=id)._get_properties()
            ' if initiator has no active paths, then remove it '
            if initiator_details['paths'][0] is None:
                LOG.info('Initiator Path does not exist.')
                host_details.delete_initiator(uid=id)
                updated_host = self.unity.get_host(name=host_details.name)
            else:
                ' Checking for initiator logged_in state '
                for path in initiator_details['paths'][0]['UnityHostInitiatorPathList']:
                    path_id = path['UnityHostInitiatorPath']['id']
                    path_id_obj = utils.host.UnityHostInitiatorPathList.get(cli=self.unity._cli, _id=path_id)
                    path_id_details = path_id_obj._get_properties()
                    " if is_logged_in is True, can't remove initiator"
                    if path_id_details['is_logged_in']:
                        error_message = 'Cannot remove initiator ' + id + ', as it is logged in the with host.'
                        LOG.error(error_message)
                        self.module.fail_json(msg=error_message)
                    elif not path_id_details['is_logged_in']:
                        ' if is_logged_in is False, remove initiator '
                        path_id_obj.delete()
                    else:
                        ' if logged_in state does not exist '
                        error_message = ' logged_in state does not exist for initiator ' + id + '.'
                        LOG.error(error_message)
                        self.module.fail_json(msg=error_message)
                host_details.delete_initiator(uid=id)
                updated_host = self.unity.get_host(name=host_details.name)
        return (True, updated_host)
    except Exception as e:
        error_message = 'Got error %s while removing initiator from host %s' % (str(e), host_details.name)
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)
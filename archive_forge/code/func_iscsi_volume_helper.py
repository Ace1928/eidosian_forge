from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
def iscsi_volume_helper(self):
    quote = dict()
    quote['iscsiInfo'] = dict()
    if self.parameters.get('igroups'):
        current_igroups = []
        for igroup in self.parameters['igroups']:
            current = self.get_igroup(igroup)
            current_igroups.append(current)
        for igroup in current_igroups:
            if igroup is None:
                quote['iscsiInfo']['igroupCreationRequest'] = dict()
                quote['iscsiInfo']['igroupCreationRequest']['igroupName'] = self.parameters['igroups'][0]
                iqn_list = []
                for initiator in self.parameters['initiators']:
                    if initiator.get('iqn'):
                        iqn_list.append(initiator['iqn'])
                        current_initiator = self.get_initiator(initiator['alias'])
                        if current_initiator is None:
                            initiator_request = dict()
                            if initiator.get('alias'):
                                initiator_request['aliasName'] = initiator['alias']
                            if initiator.get('iqn'):
                                initiator_request['iqn'] = initiator['iqn']
                            self.create_initiator(initiator_request)
                    quote['iscsiInfo']['igroupCreationRequest']['initiators'] = iqn_list
                    quote['iscsiInfo']['osName'] = self.parameters['os_name']
            else:
                quote['iscsiInfo']['igroups'] = self.parameters['igroups']
                quote['iscsiInfo']['osName'] = self.parameters['os_name']
    return quote
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
from ipaddress import ip_network
def get_interface_details(self, nas_server_obj):
    """Get interface details.
            :param: nas_server_obj: NAS server object.
            :return: Returns interface details configured on NAS server.
        """
    try:
        nas_server_obj_properties = nas_server_obj._get_properties()
        if nas_server_obj_properties['file_interface']:
            for item in nas_server_obj_properties['file_interface']['UnityFileInterfaceList']:
                interface_id = self.unity_conn.get_file_interface(_id=item['UnityFileInterface']['id'])
                if interface_id.ip_address == self.module.params['interface_ip']:
                    return interface_id
        return None
    except Exception as e:
        error_msg = 'Getting Interface details failed with error %s' % str(e)
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_cluster_dns(self):
    cluster_attrs = self.get_cluster()
    dns_attrs = None
    if not self.parameters.get('vserver') or self.parameters['vserver'] == cluster_attrs['name']:
        dns_attrs = {'domains': cluster_attrs.get('dns_domains'), 'nameservers': cluster_attrs.get('name_servers'), 'uuid': cluster_attrs['uuid']}
        self.is_cluster = True
        if dns_attrs['domains'] is None and dns_attrs['nameservers'] is None:
            dns_attrs = None
    return dns_attrs
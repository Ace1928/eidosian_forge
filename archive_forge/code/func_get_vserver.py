from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def get_vserver(svm_cx, vserver_name):
    """
    Return vserver information.

    :return:
        vserver object if vserver found
        None if vserver is not found
    :rtype: object/None
    """
    vserver_info = netapp_utils.zapi.NaElement('vserver-get-iter')
    query_details = netapp_utils.zapi.NaElement.create_node_with_children('vserver-info', **{'vserver-name': vserver_name})
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    vserver_info.add_child_elem(query)
    result = svm_cx.invoke_successfully(vserver_info, enable_tunneling=False)
    vserver_details = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attributes_list = result.get_child_by_name('attributes-list')
        vserver_info = attributes_list.get_child_by_name('vserver-info')
        aggr_list = []
        get_list = vserver_info.get_child_by_name('aggr-list')
        if get_list is not None:
            aggregates = get_list.get_children()
            aggr_list.extend((aggr.get_content() for aggr in aggregates))
        protocols = []
        allowed_protocols = vserver_info.get_child_by_name('allowed-protocols')
        if allowed_protocols is not None:
            get_protocols = allowed_protocols.get_children()
            protocols.extend((protocol.get_content() for protocol in get_protocols))
        vserver_details = {'name': vserver_info.get_child_content('vserver-name'), 'root_volume': vserver_info.get_child_content('root-volume'), 'root_volume_aggregate': vserver_info.get_child_content('root-volume-aggregate'), 'root_volume_security_style': vserver_info.get_child_content('root-volume-security-style'), 'subtype': vserver_info.get_child_content('vserver-subtype'), 'aggr_list': aggr_list, 'language': vserver_info.get_child_content('language'), 'quota_policy': vserver_info.get_child_content('quota-policy'), 'snapshot_policy': vserver_info.get_child_content('snapshot-policy'), 'allowed_protocols': protocols, 'ipspace': vserver_info.get_child_content('ipspace'), 'comment': vserver_info.get_child_content('comment'), 'max_volumes': vserver_info.get_child_content('max-volumes')}
    return vserver_details
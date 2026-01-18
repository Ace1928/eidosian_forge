from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def set_cluster_name(self, node):
    """ set up cluster name for the node using its MIP """
    cluster = dict(cluster=self.cluster_name)
    port = 442
    try:
        node_cx = netapp_utils.create_sf_connection(module=self.module, raise_on_connection_error=True, hostname=node, port=port)
    except netapp_utils.solidfire.common.ApiConnectionError as exc:
        if str(exc) == 'Bad Credentials':
            msg = 'Most likely the node %s is already in a cluster.' % node
            msg += '  Make sure to use valid node credentials for username and password.'
            msg += '  Node reported: %s' % repr(exc)
        else:
            msg = 'Failed to create connection: %s' % repr(exc)
        self.module.fail_json(msg=msg)
    except Exception as exc:
        self.module.fail_json(msg='Failed to connect to %s:%d - %s' % (node, port, to_native(exc)), exception=traceback.format_exc())
    try:
        cluster_config = node_cx.get_cluster_config()
    except netapp_utils.solidfire.common.ApiServerError as exc:
        self.module.fail_json(msg='Error getting cluster config: %s' % to_native(exc), exception=traceback.format_exc())
    if cluster_config.cluster.cluster == self.cluster_name:
        return False
    if cluster_config.cluster.state == 'Active':
        self.module.fail_json(msg="Error updating cluster name for node %s, already in 'Active' state" % node, cluster_config=repr(cluster_config))
    if self.module.check_mode:
        return True
    try:
        node_cx.set_cluster_config(cluster)
    except netapp_utils.solidfire.common.ApiServerError as exc:
        self.module.fail_json(msg='Error updating cluster name: %s' % to_native(exc), cluster_config=repr(cluster_config), exception=traceback.format_exc())
    return True
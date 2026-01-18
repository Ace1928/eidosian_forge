from ironicclient.tests.functional.osc.v1 import base
def test_deploy_rebuild_undeploy_manage(self):
    """Deploy, rebuild and undeploy node.

        Test steps:
        1) Create baremetal node in setUp.
        2) Check initial "enroll" provision state.
        3) Set baremetal node "manage" provision state.
        4) Check baremetal node provision_state field value is "manageable".
        5) Set baremetal node "provide" provision state.
        6) Check baremetal node provision_state field value is "available".
        7) Set baremetal node "deploy" provision state.
        8) Check baremetal node provision_state field value is "active".
        9) Set baremetal node "rebuild" provision state.
        10) Check baremetal node provision_state field value is "active".
        11) Set baremetal node "undeploy" provision state.
        12) Check baremetal node provision_state field value is "available".
        13) Set baremetal node "manage" provision state.
        14) Check baremetal node provision_state field value is "manageable".
        15) Set baremetal node "provide" provision state.
        16) Check baremetal node provision_state field value is "available".
        """
    show_prop = self.node_show(self.node['uuid'], ['provision_state'])
    self.assertEqual('enroll', show_prop['provision_state'])
    self.openstack('baremetal node manage {0}'.format(self.node['uuid']))
    show_prop = self.node_show(self.node['uuid'], ['provision_state'])
    self.assertEqual('manageable', show_prop['provision_state'])
    self.openstack('baremetal node provide {0}'.format(self.node['uuid']))
    show_prop = self.node_show(self.node['uuid'], ['provision_state'])
    self.assertEqual('available', show_prop['provision_state'])
    self.openstack('baremetal node deploy {0}'.format(self.node['uuid']))
    show_prop = self.node_show(self.node['uuid'], ['provision_state'])
    self.assertEqual('active', show_prop['provision_state'])
    self.openstack('baremetal node rebuild {0}'.format(self.node['uuid']))
    show_prop = self.node_show(self.node['uuid'], ['provision_state'])
    self.assertEqual('active', show_prop['provision_state'])
    self.openstack('baremetal node undeploy {0}'.format(self.node['uuid']))
    show_prop = self.node_show(self.node['uuid'], ['provision_state'])
    self.assertEqual('available', show_prop['provision_state'])
    self.openstack('baremetal node manage {0}'.format(self.node['uuid']))
    show_prop = self.node_show(self.node['uuid'], ['provision_state'])
    self.assertEqual('manageable', show_prop['provision_state'])
    self.openstack('baremetal node provide {0}'.format(self.node['uuid']))
    show_prop = self.node_show(self.node['uuid'], ['provision_state'])
    self.assertEqual('available', show_prop['provision_state'])
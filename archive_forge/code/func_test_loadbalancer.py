import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.aws.lb import loadbalancer as lb
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_loadbalancer(self):
    t = template_format.parse(lb_template)
    s = utils.parse_stack(t)
    s.store()
    resource_name = 'LoadBalancer'
    lb_defn = s.t.resource_definitions(s)[resource_name]
    rsrc = lb.LoadBalancer(resource_name, lb_defn, s)
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    initial_md = {'AWS::CloudFormation::Init': {'config': {'files': {'/etc/haproxy/haproxy.cfg': {'content': 'initial'}}}}}
    ha_cfg = '\n'.join(['\nglobal', '    daemon', '    maxconn 256', '    stats socket /tmp/.haproxy-stats', '\ndefaults', '    mode http\n    timeout connect 5000ms', '    timeout client 50000ms', '    timeout server 50000ms\n\nfrontend http', '    bind *:80\n    default_backend servers', '\nbackend servers\n    balance roundrobin', '    option http-server-close', '    option forwardfor\n    option httpchk', '\n    server server1 1.2.3.4:80', '    server server2 0.0.0.0:80\n'])
    expected_md = {'AWS::CloudFormation::Init': {'config': {'files': {'/etc/haproxy/haproxy.cfg': {'content': ha_cfg}}}}}
    md = mock.Mock()
    md.metadata_get.return_value = copy.deepcopy(initial_md)
    rsrc.nested = mock.Mock(return_value={'LB_instance': md})
    prop_diff = {'Instances': ['WikiServerOne1', 'WikiServerOne2']}
    props = copy.copy(rsrc.properties.data)
    props.update(prop_diff)
    update_defn = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    rsrc.handle_update(update_defn, {}, prop_diff)
    self.assertIsNone(rsrc.handle_update(rsrc.t, {}, {}))
    md.metadata_get.assert_called_once_with()
    md.metadata_set.assert_called_once_with(expected_md)
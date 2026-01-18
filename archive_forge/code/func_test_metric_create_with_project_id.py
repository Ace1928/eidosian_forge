from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import metrics
from monascaclient.v2_0 import shell
@mock.patch('monascaclient.osc.migration.make_client')
def test_metric_create_with_project_id(self, mc):
    mc.return_value = c = FakeV2Client()
    project_id = 'd48e63e76a5c4e05ba26a1185f31d4aa'
    raw_args = ('metric1 123 --time 1395691090 --project-id %s' % project_id).split(' ')
    name, cmd_clazz = migr.create_command_class('do_metric_create', shell)
    cmd = cmd_clazz(mock.Mock(), mock.Mock())
    parser = cmd.get_parser(name)
    parsed_args = parser.parse_args(raw_args)
    cmd.run(parsed_args)
    data = {'timestamp': 1395691090, 'name': 'metric1', 'tenant_id': project_id, 'value': 123.0}
    c.metrics.create.assert_called_once_with(**data)
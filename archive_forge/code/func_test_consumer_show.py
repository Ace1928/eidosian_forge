import copy
from openstackclient.identity.v3 import consumer
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_consumer_show(self):
    arglist = [identity_fakes.consumer_id]
    verifylist = [('consumer', identity_fakes.consumer_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.consumers_mock.get.assert_called_with(identity_fakes.consumer_id)
    collist = ('description', 'id')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.consumer_description, identity_fakes.consumer_id)
    self.assertEqual(datalist, data)
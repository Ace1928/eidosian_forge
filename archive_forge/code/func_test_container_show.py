import copy
from unittest import mock
from openstackclient.api import object_store_v1 as object_store
from openstackclient.object.v1 import container
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
def test_container_show(self, c_mock):
    c_mock.return_value = copy.deepcopy(object_fakes.CONTAINER)
    arglist = [object_fakes.container_name]
    verifylist = [('container', object_fakes.container_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {}
    c_mock.assert_called_with(container=object_fakes.container_name, **kwargs)
    collist = ('bytes', 'count', 'name')
    self.assertEqual(collist, columns)
    datalist = (object_fakes.container_bytes, object_fakes.container_count, object_fakes.container_name)
    self.assertEqual(datalist, data)
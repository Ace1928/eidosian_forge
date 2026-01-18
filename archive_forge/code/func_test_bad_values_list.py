import testtools
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
def test_bad_values_list(self):
    _specs = ['--listarg', 'list=true', 'type=str']
    self.assertRaises(exceptions.CommandError, neutronV20.parse_args_to_dict, _specs)
    _specs = ['--listarg', 'type=list']
    self.assertRaises(exceptions.CommandError, neutronV20.parse_args_to_dict, _specs)
    _specs = ['--listarg', 'type=list', 'action=clear']
    self.assertRaises(exceptions.CommandError, neutronV20.parse_args_to_dict, _specs)
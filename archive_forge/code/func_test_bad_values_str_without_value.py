import testtools
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
def test_bad_values_str_without_value(self):
    _specs = ['--strarg', 'type=str']
    ex = self.assertRaises(exceptions.CommandError, neutronV20.parse_args_to_dict, _specs)
    self.assertEqual('Invalid values_specs --strarg type=str', str(ex))
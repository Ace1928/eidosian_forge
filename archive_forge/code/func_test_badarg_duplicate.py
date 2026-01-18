import testtools
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
def test_badarg_duplicate(self):
    _specs = ['--tag=t', '--arg1', 'value1', '--arg1', 'value1']
    self.assertRaises(exceptions.CommandError, neutronV20.parse_args_to_dict, _specs)
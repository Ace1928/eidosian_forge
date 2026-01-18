import testtools
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
def test_badarg_early_type_specification(self):
    _specs = ['type=dict', 'key=value']
    self.assertRaises(exceptions.CommandError, neutronV20.parse_args_to_dict, _specs)
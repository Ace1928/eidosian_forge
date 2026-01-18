import testtools
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
def test_default_bool(self):
    _specs = ['--my_bool', '--arg1', 'value1']
    _mydict = neutronV20.parse_args_to_dict(_specs)
    self.assertTrue(_mydict['my_bool'])
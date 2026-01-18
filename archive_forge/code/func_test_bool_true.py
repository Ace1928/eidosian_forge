import testtools
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
def test_bool_true(self):
    _specs = ['--my-bool', 'type=bool', 'true', '--arg1', 'value1']
    _mydict = neutronV20.parse_args_to_dict(_specs)
    self.assertTrue(_mydict['my_bool'])
import testtools
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
def test_goodarg_with_minus_number(self):
    _specs = ['--arg1', 'value1', '-1', '-1.0']
    _mydict = neutronV20.parse_args_to_dict(_specs)
    self.assertEqual(['value1', '-1', '-1.0'], _mydict['arg1'])
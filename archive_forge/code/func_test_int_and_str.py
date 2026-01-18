import testtools
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
def test_int_and_str(self):
    _specs = ['--my-int', 'type=int', '10', '--my-str', 'type=str', 'value1']
    _mydict = neutronV20.parse_args_to_dict(_specs)
    self.assertEqual(10, _mydict['my_int'])
    self.assertEqual('value1', _mydict['my_str'])
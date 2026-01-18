import testtools
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
def test_dict_arg(self):
    _specs = ['--tag=t', '--arg1', 'type=dict', 'key1=value1,key2=value2']
    arg1 = neutronV20.parse_args_to_dict(_specs)['arg1']
    self.assertEqual('value1', arg1['key1'])
    self.assertEqual('value2', arg1['key2'])
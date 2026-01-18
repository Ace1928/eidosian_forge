from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def test_added_inline_list(self):
    import srsly.ruamel_yaml
    s1 = dedent('\n        a:\n        - b\n        - c\n        - d\n        ')
    s = 'a: [b, c, d]\n'
    data = srsly.ruamel_yaml.load(s1, Loader=srsly.ruamel_yaml.RoundTripLoader)
    val = data['a']
    val.fa.set_flow_style()
    output = srsly.ruamel_yaml.dump(data, Dumper=srsly.ruamel_yaml.RoundTripDumper)
    assert s == output
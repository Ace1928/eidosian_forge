from __future__ import print_function
import sys
import pytest  # NOQA
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
@pytest.mark.skipif(sys.version_info >= (3, 0) or platform.python_implementation() != 'CPython', reason='srsly.ruamel_yaml not available')
def test_dump_ruamel_ordereddict(self):
    from srsly.ruamel_yaml.compat import ordereddict
    import srsly.ruamel_yaml
    x = ordereddict([('a', 1), ('b', 2)])
    res = srsly.ruamel_yaml.dump(x, Dumper=srsly.ruamel_yaml.RoundTripDumper, default_flow_style=False)
    assert res == dedent('\n        !!omap\n        - a: 1\n        - b: 2\n        ')
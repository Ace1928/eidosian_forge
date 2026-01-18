from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def rt(s):
    import srsly.ruamel_yaml
    res = srsly.ruamel_yaml.dump(srsly.ruamel_yaml.load(s, Loader=srsly.ruamel_yaml.RoundTripLoader), Dumper=srsly.ruamel_yaml.RoundTripDumper)
    return res.strip() + '\n'
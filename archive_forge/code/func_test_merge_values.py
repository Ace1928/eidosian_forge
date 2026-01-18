import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_merge_values(self):
    from srsly.ruamel_yaml import safe_load
    d = safe_load(self.yaml_str)
    data = round_trip_load(self.yaml_str)
    count = 0
    for x in data[2].values():
        count += 1
        print(count, x)
    assert count == len(d[2])
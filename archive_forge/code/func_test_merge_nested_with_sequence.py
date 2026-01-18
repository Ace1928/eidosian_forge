import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_merge_nested_with_sequence(self):
    yaml = '\n        a:\n          <<: &content\n            <<: &y2\n              1: plugh\n            2: plover\n          0: xyzzy\n        b:\n          <<: [*content, *y2]\n        '
    data = round_trip(yaml)
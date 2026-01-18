import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_merge_01(self):
    data = load(self.merge_yaml)
    compare(data, self.merge_yaml)
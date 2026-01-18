import os
import pytest
import textwrap
import numpy as np
from . import util
def test_intent_in(self):
    for s in self._get_input():
        r = self.module.test_in_bytes4(s)
        expected = self._sint(s, end=4)
        assert r == expected, s
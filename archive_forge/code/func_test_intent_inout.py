import os
import pytest
import textwrap
import numpy as np
from . import util
def test_intent_inout(self):
    for s in self._get_input(intent='inout'):
        rest = self._sint(s, start=4)
        r = self.module.test_inout_bytes4(s)
        expected = self._sint(s, end=4)
        assert r == expected
        assert rest == self._sint(s, start=4)
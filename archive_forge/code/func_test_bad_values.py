from __future__ import annotations
from typing import Any
from unittest import TestCase
from traitlets import TraitError
def test_bad_values(self) -> None:
    if hasattr(self, '_bad_values'):
        for value in self._bad_values:
            try:
                self.assertRaises(TraitError, self.assign, value)
            except AssertionError:
                raise AssertionError(value) from None
import array
import datetime as dt
import pytest
from unittest import TestCase
from traitlets import HasTraits, Int, TraitError
from traitlets.tests.test_traitlets import TraitTestBase
from ipywidgets import Color, NumberFormat
from ipywidgets.widgets.widget import _remove_buffers, _put_buffers
from ipywidgets.widgets.trait_types import date_serialization, TypedTuple
def test_deserialize_date(self):
    serialized_date = {'year': 1900, 'month': 1, 'date': 18}
    expected = dt.date(1900, 2, 18)
    self.assertEqual(self.from_json(serialized_date, self.dummy_manager), expected)
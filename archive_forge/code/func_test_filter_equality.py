import unittest
from unittest import mock
from traits.api import HasTraits, Int
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._testing import (
def test_filter_equality(self):
    filter1 = MetadataFilter(metadata_name='name')
    filter2 = MetadataFilter(metadata_name='name')
    self.assertEqual(filter1, filter2)
    self.assertEqual(hash(filter1), hash(filter2))
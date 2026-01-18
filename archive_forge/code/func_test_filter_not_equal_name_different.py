import unittest
from unittest import mock
from traits.api import HasTraits, Int
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._testing import (
def test_filter_not_equal_name_different(self):
    filter1 = MetadataFilter(metadata_name='number')
    filter2 = MetadataFilter(metadata_name='name')
    self.assertNotEqual(filter1, filter2)
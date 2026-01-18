import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_camelize(self):
    data = {'bandwidth_limit': 'BandwidthLimit', 'test': 'Test', 'some__more__dashes': 'SomeMoreDashes', 'a_penguin_walks_into_a_bar': 'APenguinWalksIntoABar'}
    for s, expected in data.items():
        self.assertEqual(expected, helpers.camelize(s))
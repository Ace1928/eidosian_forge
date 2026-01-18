from unittest import mock
from glance_store import backend
from glance_store import exceptions
from glance_store.tests import base
def test_bad_metadata_not_dict(self):
    self._bad_metadata([])
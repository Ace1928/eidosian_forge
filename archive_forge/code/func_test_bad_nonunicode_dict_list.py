from unittest import mock
from glance_store import backend
from glance_store import exceptions
from glance_store.tests import base
def test_bad_nonunicode_dict_list(self):
    inner = {'key1': 'somevalue', 'key2': 'somevalue', 'k3': [1, object()]}
    m = {'topkey': inner, 'list': ['somevalue', '2'], 'u': '2'}
    self._bad_metadata(m)
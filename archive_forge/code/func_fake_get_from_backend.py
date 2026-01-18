from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
def fake_get_from_backend(self, location, offset=0, chunk_size=None, context=None):
    if UUID1 in location:
        raise Exception('not allow download from %s' % location)
    else:
        return self.data[location]
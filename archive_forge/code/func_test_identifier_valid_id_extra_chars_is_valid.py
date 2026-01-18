import time
from unittest import mock
import uuid
from pycadf import attachment
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import event
from pycadf import geolocation
from pycadf import host
from pycadf import identifier
from pycadf import measurement
from pycadf import metric
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import tag
from pycadf.tests import base
from pycadf import timestamp
@mock.patch('pycadf.identifier.warnings.warn')
def test_identifier_valid_id_extra_chars_is_valid(self, warning_mock):
    valid_ids = ['{1234567890abcdef1234567890abcdef}', '{12345678-1234-5678-1234-567812345678}', 'urn:uuid:12345678-1234-5678-1234-567812345678']
    for value in valid_ids:
        self.assertTrue(identifier.is_valid(value))
        self.assertFalse(warning_mock.called)
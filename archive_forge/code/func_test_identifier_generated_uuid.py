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
def test_identifier_generated_uuid(self, warning_mock):
    self.assertTrue(identifier.is_valid(identifier.generate_uuid()))
    self.assertFalse(warning_mock.called)
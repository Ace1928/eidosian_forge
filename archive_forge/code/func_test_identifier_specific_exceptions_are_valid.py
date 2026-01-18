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
def test_identifier_specific_exceptions_are_valid(self, warning_mock):
    for value in identifier.VALID_EXCEPTIONS:
        self.assertTrue(identifier.is_valid(value))
        self.assertFalse(warning_mock.called)
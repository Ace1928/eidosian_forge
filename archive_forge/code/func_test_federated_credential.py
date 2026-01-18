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
def test_federated_credential(self):
    cred = credential.FederatedCredential(token=identifier.generate_uuid(), type='http://docs.oasis-open.org/security/saml/v2.0', identity_provider=identifier.generate_uuid(), user=identifier.generate_uuid(), groups=[identifier.generate_uuid(), identifier.generate_uuid(), identifier.generate_uuid()])
    self.assertEqual(True, cred.is_valid())
    dict_cred = cred.as_dict()
    for key in credential.FED_CRED_KEYNAMES:
        self.assertIn(key, dict_cred)
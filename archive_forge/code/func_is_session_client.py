import requests
import uuid
from urllib import parse as urlparse
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
from keystoneclient.v3 import client
@property
def is_session_client(self):
    return self.client_type in (self.KSC_SESSION_CLIENT_TYPE, self.KSA_SESSION_CLIENT_TYPE)
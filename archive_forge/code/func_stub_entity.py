import requests
import uuid
from urllib import parse as urlparse
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
from keystoneclient.v3 import client
def stub_entity(self, method, parts=None, entity=None, id=None, **kwargs):
    if entity:
        entity = self.encode(entity)
        kwargs['json'] = entity
    if not parts:
        parts = [self.collection_key]
        if self.path_prefix:
            parts.insert(0, self.path_prefix)
    if id:
        if not parts:
            parts = []
        parts.append(id)
    self.stub_url(method, parts=parts, **kwargs)
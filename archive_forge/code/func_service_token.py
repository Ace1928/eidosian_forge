import itertools
from oslo_serialization import jsonutils
import webob
@property
def service_token(self):
    return self.headers.get('X-Service-Token')
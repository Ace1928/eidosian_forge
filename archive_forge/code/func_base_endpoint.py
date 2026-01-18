import http.client as http
from oslo_serialization import jsonutils
import webob
from glance.common import auth
from glance.common import exception
from glance.tests import utils
@property
def base_endpoint(self):
    return {'adminURL': 'http://localhost:9292', 'internalURL': 'http://localhost:9292', 'publicURL': 'http://localhost:9292'}
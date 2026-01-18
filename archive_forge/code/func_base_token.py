import http.client as http
from oslo_serialization import jsonutils
import webob
from glance.common import auth
from glance.common import exception
from glance.tests import utils
@property
def base_token(self):
    return {'access': {'token': {'expires': '2010-11-23T16:40:53.321584', 'id': '5c7f8799-2e54-43e4-851b-31f81871b6c', 'tenant': {'id': '1', 'name': 'tenant-ok'}}, 'serviceCatalog': [], 'user': {'id': '2', 'roles': [{'tenantId': '1', 'id': '1', 'name': 'Admin'}], 'name': 'joeadmin'}}}
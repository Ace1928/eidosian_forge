from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def fake_request_get_no_v3():
    versions = {'versions': [{'id': 'v2.0', 'links': [{'href': 'http://docs.openstack.org/', 'rel': 'describedby', 'type': 'text/html'}, {'href': 'http://192.168.122.197/v2/', 'rel': 'self'}], 'media-types': [{'base': 'application/json', 'type': 'application/'}], 'min_version': '', 'status': 'DEPRECATED', 'updated': '2014-06-28T12:20:21Z', 'version': ''}]}
    return versions
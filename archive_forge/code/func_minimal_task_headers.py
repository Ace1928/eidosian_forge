import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
def minimal_task_headers(owner='tenant1'):
    headers = {'X-Auth-Token': 'user1:%s:admin' % owner, 'Content-Type': 'application/json'}
    return headers
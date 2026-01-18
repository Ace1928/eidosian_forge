from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def put_os_services_enable(self, body, **kw):
    return (200, {}, {'host': body['host'], 'binary': body['binary'], 'status': 'enabled'})
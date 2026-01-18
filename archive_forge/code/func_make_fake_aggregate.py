import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_aggregate(id, name, availability_zone='nova', metadata=None, hosts=None):
    if not metadata:
        metadata = {}
    if not hosts:
        hosts = []
    return json.loads(json.dumps({'availability_zone': availability_zone, 'created_at': datetime.datetime.now().isoformat(), 'deleted': False, 'deleted_at': None, 'hosts': hosts, 'id': int(id), 'metadata': {'availability_zone': availability_zone}, 'name': name, 'updated_at': None}))
import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_nova_security_group(id, name, description, rules):
    if not rules:
        rules = []
    return json.loads(json.dumps({'id': id, 'name': name, 'description': description, 'tenant_id': PROJECT_ID, 'rules': rules}))
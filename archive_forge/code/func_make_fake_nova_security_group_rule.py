import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_nova_security_group_rule(id, from_port, to_port, ip_protocol, cidr):
    return json.loads(json.dumps({'id': id, 'from_port': int(from_port), 'to_port': int(to_port), 'ip_protcol': 'tcp', 'ip_range': {'cidr': cidr}}))
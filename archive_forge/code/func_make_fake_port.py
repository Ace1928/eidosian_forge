import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_port(address, node_id=None, port_id=None):
    if not node_id:
        node_id = uuid.uuid4().hex
    if not port_id:
        port_id = uuid.uuid4().hex
    return meta.obj_to_munch(FakeMachinePort(id=port_id, address=address, node_id=node_id))
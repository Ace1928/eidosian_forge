import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_machine(machine_name, machine_id=None):
    if not machine_id:
        machine_id = uuid.uuid4().hex
    return meta.obj_to_munch(FakeMachine(id=machine_id, name=machine_name))
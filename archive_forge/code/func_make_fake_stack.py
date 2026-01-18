import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_stack(id, name, description=None, status='CREATE_COMPLETE'):
    return {'creation_time': '2017-03-23T23:57:12Z', 'deletion_time': '2017-03-23T23:57:12Z', 'description': description, 'id': id, 'links': [], 'parent': None, 'stack_name': name, 'stack_owner': None, 'stack_status': status, 'stack_user_project_id': PROJECT_ID, 'tags': None, 'updated_time': '2017-03-23T23:57:12Z'}
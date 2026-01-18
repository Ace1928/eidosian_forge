import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_keypair(name):
    return {'fingerprint': '7e:eb:ab:24:ba:d1:e1:88:ae:9a:fb:66:53:df:d3:bd', 'name': name, 'type': 'ssh', 'public_key': FAKE_PUBLIC_KEY, 'created_at': datetime.datetime.now().isoformat()}
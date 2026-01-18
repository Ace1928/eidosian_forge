import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def set_tags_for_unset(resource_get, resource_id, attrs, clear_tags=False):
    if clear_tags:
        attrs['tags'] = []
    elif attrs.get('tags'):
        resource = resource_get(resource_id)
        tags = set(resource['tags'])
        tags -= set(attrs['tags'])
        attrs['tags'] = list(tags)
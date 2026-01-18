import datetime
import hashlib
import http.client as http
import os
import re
import urllib.parse as urlparse
import uuid
from castellan.common import exception as castellan_exception
from castellan import key_manager
import glance_store
from glance_store import location
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from oslo_utils import encodeutils
from oslo_utils import timeutils as oslo_timeutils
import requests
import webob.exc
from glance.api import common
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
from glance import context as glance_context
import glance.db
import glance.gateway
from glance.i18n import _, _LE, _LI, _LW
import glance.notifier
from glance.quota import keystone as ks_quota
import glance.schema
def get_base_properties():
    return {'id': {'type': 'string', 'description': _('An identifier for the image'), 'pattern': '^([0-9a-fA-F]){8}-([0-9a-fA-F]){4}-([0-9a-fA-F]){4}-([0-9a-fA-F]){4}-([0-9a-fA-F]){12}$'}, 'name': {'type': ['null', 'string'], 'description': _('Descriptive name for the image'), 'maxLength': 255}, 'status': {'type': 'string', 'readOnly': True, 'description': _('Status of the image'), 'enum': ['queued', 'saving', 'active', 'killed', 'deleted', 'uploading', 'importing', 'pending_delete', 'deactivated']}, 'visibility': {'type': 'string', 'description': _('Scope of image accessibility'), 'enum': ['community', 'public', 'private', 'shared']}, 'protected': {'type': 'boolean', 'description': _('If true, image will not be deletable.')}, 'os_hidden': {'type': 'boolean', 'description': _('If true, image will not appear in default image list response.')}, 'checksum': {'type': ['null', 'string'], 'readOnly': True, 'description': _('md5 hash of image contents.'), 'maxLength': 32}, 'os_hash_algo': {'type': ['null', 'string'], 'readOnly': True, 'description': _('Algorithm to calculate the os_hash_value'), 'maxLength': 64}, 'os_hash_value': {'type': ['null', 'string'], 'readOnly': True, 'description': _('Hexdigest of the image contents using the algorithm specified by the os_hash_algo'), 'maxLength': 128}, 'owner': {'type': ['null', 'string'], 'description': _('Owner of the image'), 'maxLength': 255}, 'size': {'type': ['null', 'integer'], 'readOnly': True, 'description': _('Size of image file in bytes')}, 'virtual_size': {'type': ['null', 'integer'], 'readOnly': True, 'description': _('Virtual size of image in bytes')}, 'container_format': {'type': ['null', 'string'], 'description': _('Format of the container'), 'enum': [None] + CONF.image_format.container_formats}, 'disk_format': {'type': ['null', 'string'], 'description': _('Format of the disk'), 'enum': [None] + CONF.image_format.disk_formats}, 'created_at': {'type': 'string', 'readOnly': True, 'description': _('Date and time of image registration')}, 'updated_at': {'type': 'string', 'readOnly': True, 'description': _('Date and time of the last image modification')}, 'tags': {'type': 'array', 'description': _('List of strings related to the image'), 'items': {'type': 'string', 'maxLength': 255}}, 'direct_url': {'type': 'string', 'readOnly': True, 'description': _('URL to access the image file kept in external store')}, 'min_ram': {'type': 'integer', 'description': _('Amount of ram (in MB) required to boot image.')}, 'min_disk': {'type': 'integer', 'description': _('Amount of disk space (in GB) required to boot image.')}, 'self': {'type': 'string', 'readOnly': True, 'description': _('An image self url')}, 'file': {'type': 'string', 'readOnly': True, 'description': _('An image file url')}, 'stores': {'type': 'string', 'readOnly': True, 'description': _('Store in which image data resides.  Only present when the operator has enabled multiple stores.  May be a comma-separated list of store identifiers.')}, 'schema': {'type': 'string', 'readOnly': True, 'description': _('An image schema url')}, 'locations': {'type': 'array', 'items': {'type': 'object', 'properties': {'url': {'type': 'string', 'maxLength': 255}, 'metadata': {'type': 'object'}, 'validation_data': {'description': _("Values to be used to populate the corresponding image properties. If the image status is not 'queued', values must exactly match those already contained in the image properties."), 'type': 'object', 'writeOnly': True, 'additionalProperties': False, 'properties': {'checksum': {'type': 'string', 'minLength': 32, 'maxLength': 32}, 'os_hash_algo': {'type': 'string', 'maxLength': 64}, 'os_hash_value': {'type': 'string', 'maxLength': 128}}, 'required': ['os_hash_algo', 'os_hash_value']}}, 'required': ['url', 'metadata']}, 'description': _('A set of URLs to access the image file kept in external store')}}
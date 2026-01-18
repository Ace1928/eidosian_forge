import http.client as http
import urllib.parse as urlparse
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2.model.metadef_namespace import Namespace
from glance.api.v2.model.metadef_namespace import Namespaces
from glance.api.v2.model.metadef_object import MetadefObject
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.api.v2.model.metadef_resource_type import ResourceTypeAssociation
from glance.api.v2.model.metadef_tag import MetadefTag
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance.common import wsme_utils
import glance.db
import glance.gateway
from glance.i18n import _, _LE
import glance.notifier
import glance.schema
def get_schema_definitions():
    return {'positiveInteger': {'type': 'integer', 'minimum': 0}, 'positiveIntegerDefault0': {'allOf': [{'$ref': '#/definitions/positiveInteger'}, {'default': 0}]}, 'stringArray': {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True}, 'property': {'type': 'object', 'additionalProperties': {'type': 'object', 'required': ['title', 'type'], 'properties': {'name': {'type': 'string', 'maxLength': 80}, 'title': {'type': 'string'}, 'description': {'type': 'string'}, 'operators': {'type': 'array', 'items': {'type': 'string'}}, 'type': {'type': 'string', 'enum': ['array', 'boolean', 'integer', 'number', 'object', 'string', None]}, 'required': {'$ref': '#/definitions/stringArray'}, 'minimum': {'type': 'number'}, 'maximum': {'type': 'number'}, 'maxLength': {'$ref': '#/definitions/positiveInteger'}, 'minLength': {'$ref': '#/definitions/positiveIntegerDefault0'}, 'pattern': {'type': 'string', 'format': 'regex'}, 'enum': {'type': 'array'}, 'readonly': {'type': 'boolean'}, 'default': {}, 'items': {'type': 'object', 'properties': {'type': {'type': 'string', 'enum': ['array', 'boolean', 'integer', 'number', 'object', 'string', None]}, 'enum': {'type': 'array'}}}, 'maxItems': {'$ref': '#/definitions/positiveInteger'}, 'minItems': {'$ref': '#/definitions/positiveIntegerDefault0'}, 'uniqueItems': {'type': 'boolean', 'default': False}, 'additionalItems': {'type': 'boolean'}}}}}
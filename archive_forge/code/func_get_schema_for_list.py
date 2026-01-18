import http.client as http
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2.model.metadef_tag import MetadefTag
from glance.api.v2.model.metadef_tag import MetadefTags
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
from glance.common import wsme_utils
import glance.db
from glance.i18n import _
import glance.notifier
import glance.schema
def get_schema_for_list():
    definitions = _get_base_definitions()
    properties = _get_base_properties_for_list()
    schema = glance.schema.Schema('tags', properties, required=None, definitions=definitions)
    return schema
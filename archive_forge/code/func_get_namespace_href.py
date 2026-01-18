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
def get_namespace_href(namespace):
    base_href = '/v2/metadefs/namespaces/%s' % namespace.namespace
    return base_href
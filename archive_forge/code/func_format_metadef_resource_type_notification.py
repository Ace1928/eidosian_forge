import abc
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob
from glance.common import exception
from glance.common import timeutils
from glance.domain import proxy as domain_proxy
from glance.i18n import _, _LE
def format_metadef_resource_type_notification(metadef_resource_type):
    return {'namespace': metadef_resource_type.namespace, 'name': metadef_resource_type.name, 'name_old': metadef_resource_type.name, 'prefix': metadef_resource_type.prefix, 'properties_target': metadef_resource_type.properties_target, 'created_at': timeutils.isotime(metadef_resource_type.created_at), 'updated_at': timeutils.isotime(metadef_resource_type.updated_at), 'deleted': False, 'deleted_at': None}
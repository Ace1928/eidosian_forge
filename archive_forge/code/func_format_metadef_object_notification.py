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
def format_metadef_object_notification(metadef_object):
    object_properties = metadef_object.properties or {}
    properties = []
    for name, prop in object_properties.items():
        object_property = _format_metadef_object_property(name, prop)
        properties.append(object_property)
    return {'namespace': metadef_object.namespace, 'name': metadef_object.name, 'name_old': metadef_object.name, 'properties': properties, 'required': metadef_object.required, 'description': metadef_object.description, 'created_at': timeutils.isotime(metadef_object.created_at), 'updated_at': timeutils.isotime(metadef_object.updated_at), 'deleted': False, 'deleted_at': None}
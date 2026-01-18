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
def format_metadef_property_notification(metadef_property):
    schema = metadef_property.schema
    return {'namespace': metadef_property.namespace, 'name': metadef_property.name, 'name_old': metadef_property.name, 'type': schema.get('type'), 'title': schema.get('title'), 'description': schema.get('description'), 'default': schema.get('default'), 'minimum': schema.get('minimum'), 'maximum': schema.get('maximum'), 'enum': schema.get('enum'), 'pattern': schema.get('pattern'), 'minLength': schema.get('minLength'), 'maxLength': schema.get('maxLength'), 'confidential': schema.get('confidential'), 'items': schema.get('items'), 'uniqueItems': schema.get('uniqueItems'), 'minItems': schema.get('minItems'), 'maxItems': schema.get('maxItems'), 'additionalItems': schema.get('additionalItems'), 'deleted': False, 'deleted_at': None}
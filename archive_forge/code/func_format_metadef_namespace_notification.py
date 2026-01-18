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
def format_metadef_namespace_notification(metadef_namespace):
    return {'namespace': metadef_namespace.namespace, 'namespace_old': metadef_namespace.namespace, 'display_name': metadef_namespace.display_name, 'protected': metadef_namespace.protected, 'visibility': metadef_namespace.visibility, 'owner': metadef_namespace.owner, 'description': metadef_namespace.description, 'created_at': timeutils.isotime(metadef_namespace.created_at), 'updated_at': timeutils.isotime(metadef_namespace.updated_at), 'deleted': False, 'deleted_at': None}
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
def format_metadef_tag_notification(metadef_tag):
    return {'namespace': metadef_tag.namespace, 'name': metadef_tag.name, 'name_old': metadef_tag.name, 'created_at': timeutils.isotime(metadef_tag.created_at), 'updated_at': timeutils.isotime(metadef_tag.updated_at), 'deleted': False, 'deleted_at': None}
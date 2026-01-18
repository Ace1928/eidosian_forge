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
def format_task_notification(task):
    return {'id': task.task_id, 'type': task.type, 'status': task.status, 'result': None, 'owner': task.owner, 'message': None, 'expires_at': timeutils.isotime(task.expires_at), 'created_at': timeutils.isotime(task.created_at), 'updated_at': timeutils.isotime(task.updated_at), 'deleted': False, 'deleted_at': None}
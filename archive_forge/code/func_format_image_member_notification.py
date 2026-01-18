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
def format_image_member_notification(image_member):
    """Given a glance.domain.ImageMember object, return a dictionary of
    relevant notification information.
    """
    return {'image_id': image_member.image_id, 'member_id': image_member.member_id, 'status': image_member.status, 'created_at': timeutils.isotime(image_member.created_at), 'updated_at': timeutils.isotime(image_member.updated_at), 'deleted': False, 'deleted_at': None}
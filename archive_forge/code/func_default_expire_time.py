import datetime
from oslo_log import log
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import manager
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import receipt_model
from keystone import notifications
def default_expire_time():
    """Determine when a fresh receipt should expire.

    Expiration time varies based on configuration (see
    ``[receipt] expiration``).

    :returns: a naive UTC datetime.datetime object

    """
    expire_delta = datetime.timedelta(seconds=CONF.receipt.expiration)
    expires_at = timeutils.utcnow() + expire_delta
    return expires_at.replace(microsecond=0)
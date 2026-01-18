from oslo_config import cfg
from heat.common.i18n import _
from heat.common import messaging
def get_default_level():
    return CONF.default_notification_level.upper()
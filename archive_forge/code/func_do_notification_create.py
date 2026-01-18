import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('name', metavar='<NOTIFICATION_NAME>', help='Name of the notification to create.')
@utils.arg('type', metavar='<TYPE>', help='The notification type. See monasca notification-type-list for supported types.')
@utils.arg('address', metavar='<ADDRESS>', help='A valid EMAIL Address, URL, or SERVICE KEY.')
@utils.arg('--period', metavar='<PERIOD>', type=int, default=0, help='A period for the notification method.')
def do_notification_create(mc, args):
    """Create notification."""
    fields = {}
    fields['name'] = args.name
    fields['type'] = args.type
    fields['address'] = args.address
    if args.period:
        fields['period'] = args.period
    try:
        notification = mc.notifications.create(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        print(jsonutils.dumps(notification, indent=2))
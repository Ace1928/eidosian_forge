import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('id', metavar='<NOTIFICATION_ID>', help='The ID of the notification.')
@utils.arg('--name', metavar='<NOTIFICATION_NAME>', help='Name of the notification.')
@utils.arg('--type', metavar='<TYPE>', help='The notification type. See monasca notification-type-list for supported types.')
@utils.arg('--address', metavar='<ADDRESS>', help='A valid EMAIL Address, URL, or SERVICE KEY.')
@utils.arg('--period', metavar='<PERIOD>', type=int, help='A period for the notification method.')
def do_notification_patch(mc, args):
    """Patch notification."""
    fields = {}
    fields['notification_id'] = args.id
    if args.name:
        fields['name'] = args.name
    if args.type:
        fields['type'] = args.type
    if args.address:
        fields['address'] = args.address
    if args.period or args.period == 0:
        fields['period'] = args.period
    try:
        notification = mc.notifications.patch(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        print(jsonutils.dumps(notification, indent=2))
import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('id', metavar='<NOTIFICATION_ID>', help='The ID of the notification.')
def do_notification_delete(mc, args):
    """Delete notification."""
    fields = {}
    fields['notification_id'] = args.id
    try:
        mc.notifications.delete(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        print('Successfully deleted notification')
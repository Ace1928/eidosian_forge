import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def do_notification_type_list(mc, args):
    """List notification types supported by monasca."""
    try:
        notification_types = mc.notificationtypes.list()
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        if args.json:
            print(utils.json_formatter(notification_types))
            return
        else:
            formatters = {'types': lambda x: x['type']}
            utils.print_list(notification_types, ['types'], formatters=formatters)
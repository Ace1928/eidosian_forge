import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('Id or Name of the instance.'))
@utils.arg('log_name', metavar='<log_name>', help=_('Name of log to show.'))
@utils.service_type('database')
def do_log_show(cs, args):
    """Instructs Trove guest to show details of log."""
    try:
        instance = _find_instance(cs, args.instance)
        log_info = cs.instances.log_show(instance, args.log_name)
        _print_object(log_info)
    except exceptions.GuestLogNotFoundError:
        print(NO_LOG_FOUND_ERROR % {'log_name': args.log_name, 'instance': instance})
    except Exception as ex:
        error_msg = ex.message.split('\n')
        print(error_msg[0])
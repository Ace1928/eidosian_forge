import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('Id or Name of the instance.'))
@utils.arg('log_name', metavar='<log_name>', help=_('Name of log to publish.'))
@utils.arg('--file', metavar='<file>', default=None, help=_('Path of file to save log to for instance.'))
@utils.service_type('database')
def do_log_save(cs, args):
    """Save log file for instance."""
    try:
        instance = _find_instance(cs, args.instance)
        filename = cs.instances.log_save(instance, args.log_name, filename=args.file)
        print(_('Log "%(log_name)s" written to %(file_name)s') % {'log_name': args.log_name, 'file_name': filename})
    except exceptions.GuestLogNotFoundError:
        print(NO_LOG_FOUND_ERROR % {'log_name': args.log_name, 'instance': instance})
    except Exception as ex:
        error_msg = ex.message.split('\n')
        print(error_msg[0])
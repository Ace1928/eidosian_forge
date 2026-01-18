from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('host', metavar='<host>', help='Name of host.')
@utils.arg('binary', metavar='<binary>', help='Name of the binary to delete.')
def do_service_delete(cs, args):
    """Delete the Zun binaries/services."""
    try:
        cs.services.delete(args.host, args.binary)
        print('Request to delete binary %s on host %s has been accepted.' % (args.binary, args.host))
    except Exception as e:
        print('Delete for binary %(binary)s on host %(host)s failed: %(e)s' % {'binary': args.binary, 'host': args.host, 'e': e})
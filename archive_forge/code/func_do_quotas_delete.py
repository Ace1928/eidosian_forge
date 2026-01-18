from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
@utils.arg('--project-id', required=True, metavar='<project-id>', help=_('Project ID.'))
@utils.arg('--resource', required=True, metavar='<resource>', help=_('Resource name'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_quotas_delete(cs, args):
    """Delete specified resource quota."""
    try:
        cs.quotas.delete(args.project_id, args.resource)
        print('Request to delete quota for project id %(id)s and resource %(res)s has been accepted.' % {'id': args.project_id, 'res': args.resource})
    except Exception as e:
        print('Quota delete failed for project id %(id)s and resource %(res)s :%(e)s' % {'id': args.project_id, 'res': args.resource, 'e': e.details})
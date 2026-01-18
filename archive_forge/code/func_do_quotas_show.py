from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
@utils.arg('--project-id', required=True, metavar='<project-id>', help=_('Project ID.'))
@utils.arg('--resource', required=True, metavar='<resource>', help=_('Resource name'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_quotas_show(cs, args):
    """Show details about the given project resource quota."""
    quota = cs.quotas.get(args.project_id, args.resource)
    _show_quota(quota)
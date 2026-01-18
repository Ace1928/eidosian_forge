from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
@utils.arg('--project-id', required=True, metavar='<project-id>', help=_('Project Id.'))
@utils.arg('--resource', required=True, metavar='<resource>', help=_('Resource name.'))
@utils.arg('--hard-limit', metavar='<hard-limit>', type=int, default=1, help=_('Max resource limit.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_quotas_update(cs, args):
    """Update information about the given project resource quota."""
    patch = dict()
    patch['project_id'] = args.project_id
    patch['resource'] = args.resource
    patch['hard_limit'] = args.hard_limit
    quota = cs.quotas.update(args.project_id, args.resource, patch)
    _show_quota(quota)
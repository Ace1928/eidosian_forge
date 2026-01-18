from zunclient.common import cliutils as utils
@utils.arg('project_id', metavar='<project_id>', help='The UUID of project in a multi-project cloud')
def do_quota_defaults(cs, args):
    """Print a  default quotas for a project"""
    utils.print_dict(cs.quotas.defaults(args.project_id)._info)
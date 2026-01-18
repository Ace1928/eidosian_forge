from zunclient.common import cliutils as utils
@utils.arg('--usages', default=False, action='store_true', help='Whether show quota usage statistic or not')
@utils.arg('project_id', metavar='<project_id>', help='The UUID of project in a multi-project cloud')
def do_quota_get(cs, args):
    """Print a quotas for a project with usages (optional)"""
    if args.usages:
        utils.print_dict(cs.quotas.get(args.project_id, usages=args.usages)._info, value_fields=('limit', 'in_use'))
    else:
        utils.print_dict(cs.quotas.get(args.project_id, usages=args.usages)._info)
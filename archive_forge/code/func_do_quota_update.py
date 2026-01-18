from zunclient.common import cliutils as utils
@utils.arg('--containers', metavar='<containers>', type=int, help='The number of containers allowed per project')
@utils.arg('--cpu', metavar='<cpu>', type=int, help='The number of container cores or vCPUs allowed per project')
@utils.arg('--memory', metavar='<memory>', type=int, help='The number of megabytes of container RAM allowed per project')
@utils.arg('--disk', metavar='<disk>', type=int, help='The number of gigabytes of container Disk allowed per project')
@utils.arg('project_id', metavar='<project_id>', help='The UUID of project in a multi-project cloud')
def do_quota_update(cs, args):
    """Print an updated quotas for a project"""
    utils.print_dict(cs.quotas.update(args.project_id, containers=args.containers, memory=args.memory, cpu=args.cpu, disk=args.disk)._info)
from zunclient.common import cliutils as utils
@utils.arg('--containers', metavar='<containers>', type=int, help='The number of containers allowed per project')
@utils.arg('--cpu', metavar='<cpu>', type=int, help='The number of container cores or vCPUs allowed per project')
@utils.arg('--memory', metavar='<memory>', type=int, help='The number of megabytes of container RAM allowed per project')
@utils.arg('--disk', metavar='<disk>', type=int, help='The number of gigabytes of container Disk allowed per project')
@utils.arg('quota_class_name', metavar='<quota_class_name>', help='The name of quota class')
def do_quota_class_update(cs, args):
    """Print an updated quotas for a quota class"""
    utils.print_dict(cs.quota_classes.update(args.quota_class_name, containers=args.containers, memory=args.memory, cpu=args.cpu, disk=args.disk)._info)
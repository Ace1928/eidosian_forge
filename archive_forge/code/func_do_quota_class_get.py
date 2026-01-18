from zunclient.common import cliutils as utils
@utils.arg('quota_class_name', metavar='<quota_class_name>', help='The name of quota class')
def do_quota_class_get(cs, args):
    """Print a quotas for a quota class"""
    utils.print_dict(cs.quota_classes.get(args.quota_class_name)._info)
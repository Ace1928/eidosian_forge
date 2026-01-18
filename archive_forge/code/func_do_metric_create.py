import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('name', metavar='<METRIC_NAME>', help='Name of the metric to create.')
@utils.arg('--dimensions', metavar='<KEY1=VALUE1,KEY2=VALUE2...>', help='key value pair used to create a metric dimension. This can be specified multiple times, or once with parameters separated by a comma. Dimensions need quoting when they contain special chars [&,(,),{,},>,<] that confuse the CLI parser.', action='append')
@utils.arg('--value-meta', metavar='<KEY1=VALUE1,KEY2=VALUE2...>', help='key value pair for extra information about a value. This can be specified multiple times, or once with parameters separated by a comma. value_meta need quoting when they contain special chars [&,(,),{,},>,<] that confuse the CLI parser.', action='append')
@utils.arg('--time', metavar='<UNIX_TIMESTAMP>', default=time.time() * 1000, type=int, help='Metric timestamp in milliseconds. Default: current timestamp.')
@utils.arg('--project-id', metavar='<CROSS_PROJECT_ID>', help='The Project ID to create metric on behalf of. Requires monitoring-delegate role in keystone.')
@utils.arg('value', metavar='<METRIC_VALUE>', type=float, help='Metric value.')
def do_metric_create(mc, args):
    """Create metric."""
    fields = {}
    fields['name'] = args.name
    if args.dimensions:
        fields['dimensions'] = utils.format_parameters(args.dimensions)
    fields['timestamp'] = args.time
    fields['value'] = args.value
    if args.value_meta:
        fields['value_meta'] = utils.format_parameters(args.value_meta)
    if args.project_id:
        fields['tenant_id'] = args.project_id
    try:
        mc.metrics.create(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        print('Successfully created metric')
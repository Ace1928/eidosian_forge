import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('--metric-name', metavar='<METRIC_NAME>', help='Name of the metric to report dimension name list.', action='append')
@utils.arg('--offset', metavar='<OFFSET LOCATION>', help='The offset used to paginate the return data.')
@utils.arg('--limit', metavar='<RETURN LIMIT>', help='The amount of data to be returned up to the API maximum limit.')
@utils.arg('--tenant-id', metavar='<TENANT_ID>', help="Retrieve data for the specified tenant/project id instead of the tenant/project from the user's Keystone credentials.")
def do_dimension_name_list(mc, args):
    """List names of metric dimensions."""
    fields = {}
    if args.metric_name:
        fields['metric_name'] = args.metric_name
    if args.limit:
        fields['limit'] = args.limit
    if args.offset:
        fields['offset'] = args.offset
    if args.tenant_id:
        fields['tenant_id'] = args.tenant_id
    try:
        dimension_names = mc.metrics.list_dimension_names(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    if args.json:
        print(utils.json_formatter(dimension_names))
        return
    if isinstance(dimension_names, list):
        utils.print_list(dimension_names, ['Dimension Names'], formatters={'Dimension Names': lambda x: x['dimension_name']})
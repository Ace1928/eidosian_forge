import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('dimension_name', metavar='<DIMENSION_NAME>', help='Name of the dimension to list dimension values.')
@utils.arg('--metric-name', metavar='<METRIC_NAME>', help='Name of the metric to report dimension value list.', action='append')
@utils.arg('--offset', metavar='<OFFSET LOCATION>', help='The offset used to paginate the return data.')
@utils.arg('--limit', metavar='<RETURN LIMIT>', help='The amount of data to be returned up to the API maximum limit.')
@utils.arg('--tenant-id', metavar='<TENANT_ID>', help="Retrieve data for the specified tenant/project id instead of the tenant/project from the user's Keystone credentials.")
def do_dimension_value_list(mc, args):
    """List names of metric dimensions."""
    fields = {}
    fields['dimension_name'] = args.dimension_name
    if args.metric_name:
        fields['metric_name'] = args.metric_name
    if args.limit:
        fields['limit'] = args.limit
    if args.offset:
        fields['offset'] = args.offset
    if args.tenant_id:
        fields['tenant_id'] = args.tenant_id
    try:
        dimension_values = mc.metrics.list_dimension_values(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    if args.json:
        print(utils.json_formatter(dimension_values))
        return
    if isinstance(dimension_values, list):
        utils.print_list(dimension_values, ['Dimension Values'], formatters={'Dimension Values': lambda x: x['dimension_value']})
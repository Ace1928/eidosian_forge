import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('name', metavar='<METRIC_NAME>', help='Name of the metric to report measurement statistics.')
@utils.arg('statistics', metavar='<STATISTICS>', help='Statistics is one or more (separated by commas) of [AVG, MIN, MAX, COUNT, SUM].')
@utils.arg('--dimensions', metavar='<KEY1=VALUE1,KEY2=VALUE2...>', help='key value pair used to specify a metric dimension. This can be specified multiple times, or once with parameters separated by a comma. Dimensions need quoting when they contain special chars [&,(,),{,},>,<] that confuse the CLI parser.', action='append')
@utils.arg('starttime', metavar='<UTC_START_TIME>', help='measurements >= UTC time. format: 2014-01-01T00:00:00Z. OR Format: -120 (previous 120 minutes).')
@utils.arg('--endtime', metavar='<UTC_END_TIME>', help='measurements <= UTC time. format: 2014-01-01T00:00:00Z.')
@utils.arg('--period', metavar='<PERIOD>', help='number of seconds per interval (default is 300)')
@utils.arg('--offset', metavar='<OFFSET LOCATION>', help='The offset used to paginate the return data.')
@utils.arg('--limit', metavar='<RETURN LIMIT>', help='The amount of data to be returned up to the API maximum limit.')
@utils.arg('--merge_metrics', action='store_const', const=True, help='Merge multiple metrics into a single result.')
@utils.arg('--group_by', metavar='<KEY1,KEY2,...>', help="Select which keys to use for grouping. A '*' groups by all keys.")
@utils.arg('--tenant-id', metavar='<TENANT_ID>', help="Retrieve data for the specified tenant/project id instead of the tenant/project from the user's Keystone credentials.")
def do_metric_statistics(mc, args):
    """List measurement statistics for the specified metric."""
    statistic_types = ['AVG', 'MIN', 'MAX', 'COUNT', 'SUM']
    statlist = args.statistics.split(',')
    for stat in statlist:
        if stat.upper() not in statistic_types:
            errmsg = 'Invalid type, not one of [' + ', '.join(statistic_types) + ']'
            raise osc_exc.CommandError(errmsg)
    fields = {}
    fields['name'] = args.name
    if args.dimensions:
        fields['dimensions'] = utils.format_dimensions_query(args.dimensions)
    _translate_starttime(args)
    fields['start_time'] = args.starttime
    if args.endtime:
        fields['end_time'] = args.endtime
    if args.period:
        fields['period'] = args.period
    fields['statistics'] = args.statistics
    if args.limit:
        fields['limit'] = args.limit
    if args.offset:
        fields['offset'] = args.offset
    if args.merge_metrics:
        fields['merge_metrics'] = args.merge_metrics
    if args.group_by:
        fields['group_by'] = args.group_by
    if args.tenant_id:
        fields['tenant_id'] = args.tenant_id
    try:
        metric = mc.metrics.list_statistics(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        if args.json:
            print(utils.json_formatter(metric))
            return
        cols = ['name', 'dimensions']
        if metric:
            column_names = metric[0]['columns']
            for name in column_names:
                cols.append(name)
        else:
            cols.append('timestamp')
        formatters = {'name': lambda x: x['name'], 'dimensions': lambda x: utils.format_dict(x['dimensions']), 'timestamp': lambda x: format_statistic_timestamp(x['statistics'], x['columns'], 'timestamp'), 'avg': lambda x: format_statistic_value(x['statistics'], x['columns'], 'avg'), 'min': lambda x: format_statistic_value(x['statistics'], x['columns'], 'min'), 'max': lambda x: format_statistic_value(x['statistics'], x['columns'], 'max'), 'count': lambda x: format_statistic_value(x['statistics'], x['columns'], 'count'), 'sum': lambda x: format_statistic_value(x['statistics'], x['columns'], 'sum')}
        if isinstance(metric, list):
            utils.print_list(metric, cols, formatters=formatters)
        else:
            metric_list = list()
            metric_list.append(metric)
            utils.print_list(metric_list, cols, formatters=formatters)
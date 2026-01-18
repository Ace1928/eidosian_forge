import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('--alarm-definition-id', metavar='<ALARM_DEFINITION_ID>', help='The ID of the alarm definition.')
@utils.arg('--metric-name', metavar='<METRIC_NAME>', help='Name of the metric.')
@utils.arg('--metric-dimensions', metavar='<KEY1=VALUE1,KEY2,KEY3=VALUE2...>', help='key value pair used to specify a metric dimension or just key to select all values of that dimension.This can be specified multiple times, or once with parameters separated by a comma. Dimensions need quoting when they contain special chars [&,(,),{,},>,<] that confuse the CLI parser.', action='append')
@utils.arg('--state', metavar='<ALARM_STATE>', help='ALARM_STATE is one of [UNDETERMINED, OK, ALARM].')
@utils.arg('--severity', metavar='<SEVERITY>', help='Severity is one of ["LOW", "MEDIUM", "HIGH", "CRITICAL"].')
@utils.arg('--state-updated-start-time', metavar='<UTC_STATE_UPDATED_START>', help='Return all alarms whose state was updated on or after the time specified.')
@utils.arg('--lifecycle-state', metavar='<LIFECYCLE_STATE>', help='The lifecycle state of the alarm.')
@utils.arg('--link', metavar='<LINK>', help='The link to external data associated with the alarm.')
@utils.arg('--group-by', metavar='<GROUP_BY>', help='Comma separated list of one or more fields to group the results by. Group by is one or more of [alarm_definition_id, name, state, link, lifecycle_state, metric_name, dimension_name, dimension_value].')
@utils.arg('--offset', metavar='<OFFSET LOCATION>', help='The offset used to paginate the return data.')
@utils.arg('--limit', metavar='<RETURN LIMIT>', help='The amount of data to be returned up to the API maximum limit.')
def do_alarm_count(mc, args):
    """Count alarms."""
    fields = {}
    if args.alarm_definition_id:
        fields['alarm_definition_id'] = args.alarm_definition_id
    if args.metric_name:
        fields['metric_name'] = args.metric_name
    if args.metric_dimensions:
        fields['metric_dimensions'] = utils.format_dimensions_query(args.metric_dimensions)
    if args.state:
        if args.state.upper() not in state_types:
            errmsg = 'Invalid state, not one of [' + ', '.join(state_types) + ']'
            print(errmsg)
            return
        fields['state'] = args.state
    if args.severity:
        if not _validate_severity(args.severity):
            return
        fields['severity'] = args.severity
    if args.state_updated_start_time:
        fields['state_updated_start_time'] = args.state_updated_start_time
    if args.lifecycle_state:
        fields['lifecycle_state'] = args.lifecycle_state
    if args.link:
        fields['link'] = args.link
    if args.group_by:
        group_by = args.group_by.split(',')
        if not set(group_by).issubset(set(group_by_types)):
            errmsg = 'Invalid group-by, one or more values not in [' + ','.join(group_by_types) + ']'
            print(errmsg)
            return
        fields['group_by'] = args.group_by
    if args.limit:
        fields['limit'] = args.limit
    if args.offset:
        fields['offset'] = args.offset
    try:
        counts = mc.alarms.count(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        if args.json:
            print(utils.json_formatter(counts))
            return
        cols = counts['columns']
        utils.print_list(counts['counts'], [i for i in range(len(cols))], field_labels=cols)
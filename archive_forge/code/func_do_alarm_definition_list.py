import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('--name', metavar='<ALARM_DEFINITION_NAME>', help='Name of the alarm definition.')
@utils.arg('--dimensions', metavar='<KEY1=VALUE1,KEY2=VALUE2...>', help='key value pair used to specify a metric dimension. This can be specified multiple times, or once with parameters separated by a comma. Dimensions need quoting when they contain special chars [&,(,),{,},>,<] that confuse the CLI parser.', action='append')
@utils.arg('--severity', metavar='<SEVERITY>', help='Severity is one of ["LOW", "MEDIUM", "HIGH", "CRITICAL"].')
@utils.arg('--sort-by', metavar='<SORT BY FIELDS>', help='Fields to sort by as a comma separated list. Valid values are id, name, severity, created_at, updated_at. Fields may be followed by "asc" or "desc", ex "severity desc", to set the direction of sorting.')
@utils.arg('--offset', metavar='<OFFSET LOCATION>', help='The offset used to paginate the return data.')
@utils.arg('--limit', metavar='<RETURN LIMIT>', help='The amount of data to be returned up to the API maximum limit.')
def do_alarm_definition_list(mc, args):
    """List alarm definitions for this tenant."""
    fields = {}
    if args.name:
        fields['name'] = args.name
    if args.dimensions:
        fields['dimensions'] = utils.format_dimensions_query(args.dimensions)
    if args.severity:
        if not _validate_severity(args.severity):
            return
        fields['severity'] = args.severity
    if args.sort_by:
        sort_by = args.sort_by.split(',')
        for field in sort_by:
            field_values = field.split()
            if len(field_values) > 2:
                print('Invalid sort_by value {}'.format(field))
            if field_values[0] not in allowed_definition_sort_by:
                print('Sort-by field name {} is not in [{}]'.format(field_values[0], allowed_definition_sort_by))
                return
            if len(field_values) > 1 and field_values[1] not in ['asc', 'desc']:
                print('Invalid value {}, must be asc or desc'.format(field_values[1]))
        fields['sort_by'] = args.sort_by
    if args.limit:
        fields['limit'] = args.limit
    if args.offset:
        fields['offset'] = args.offset
    try:
        alarm = mc.alarm_definitions.list(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        if args.json:
            print(utils.json_formatter(alarm))
            return
        cols = ['name', 'id', 'expression', 'match_by', 'actions_enabled']
        formatters = {'name': lambda x: x['name'], 'id': lambda x: x['id'], 'expression': lambda x: x['expression'], 'match_by': lambda x: utils.format_list(x['match_by']), 'actions_enabled': lambda x: x['actions_enabled']}
        if isinstance(alarm, list):
            utils.print_list(alarm, cols, formatters=formatters)
        else:
            alarm_list = list()
            alarm_list.append(alarm)
            utils.print_list(alarm_list, cols, formatters=formatters)
import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('--dimensions', metavar='<KEY1=VALUE1,KEY2=VALUE2...>', help='key value pair used to specify a metric dimension. This can be specified multiple times, or once with parameters separated by a comma. Dimensions need quoting when they contain special chars [&,(,),{,},>,<] that confuse the CLI parser.', action='append')
@utils.arg('--starttime', metavar='<UTC_START_TIME>', help='measurements >= UTC time. format: 2014-01-01T00:00:00Z. OR format: -120 (previous 120 minutes).')
@utils.arg('--endtime', metavar='<UTC_END_TIME>', help='measurements <= UTC time. format: 2014-01-01T00:00:00Z.')
@utils.arg('--offset', metavar='<OFFSET LOCATION>', help='The offset used to paginate the return data.')
@utils.arg('--limit', metavar='<RETURN LIMIT>', help='The amount of data to be returned up to the API maximum limit.')
def do_alarm_history_list(mc, args):
    """List alarms state history."""
    fields = {}
    if args.dimensions:
        fields['dimensions'] = utils.format_parameters(args.dimensions)
    if args.starttime:
        _translate_starttime(args)
        fields['start_time'] = args.starttime
    if args.endtime:
        fields['end_time'] = args.endtime
    if args.limit:
        fields['limit'] = args.limit
    if args.offset:
        fields['offset'] = args.offset
    try:
        alarm = mc.alarms.history_list(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        output_alarm_history(args, alarm)
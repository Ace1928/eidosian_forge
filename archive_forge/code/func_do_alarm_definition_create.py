import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('name', metavar='<ALARM_DEFINITION_NAME>', help='Name of the alarm definition to create.')
@utils.arg('--description', metavar='<DESCRIPTION>', help='Description of the alarm.')
@utils.arg('expression', metavar='<EXPRESSION>', help='The alarm expression to evaluate. Quoted.')
@utils.arg('--severity', metavar='<SEVERITY>', help='Severity is one of [LOW, MEDIUM, HIGH, CRITICAL].')
@utils.arg('--match-by', metavar='<MATCH_BY_DIMENSION_KEY1,MATCH_BY_DIMENSION_KEY2,...>', help='The metric dimensions to use to create unique alarms. One or more dimension key names separated by a comma. Key names need quoting when they contain special chars [&,(,),{,},>,<] that confuse the CLI parser.')
@utils.arg('--alarm-actions', metavar='<NOTIFICATION-ID>', help='The notification method to use when an alarm state is ALARM. This param may be specified multiple times.', action='append')
@utils.arg('--ok-actions', metavar='<NOTIFICATION-ID>', help='The notification method to use when an alarm state is OK. This param may be specified multiple times.', action='append')
@utils.arg('--undetermined-actions', metavar='<NOTIFICATION-ID>', help='The notification method to use when an alarm state is UNDETERMINED. This param may be specified multiple times.', action='append')
def do_alarm_definition_create(mc, args):
    """Create an alarm definition."""
    fields = {}
    fields['name'] = args.name
    if args.description:
        fields['description'] = args.description
    fields['expression'] = args.expression
    if args.alarm_actions:
        fields['alarm_actions'] = args.alarm_actions
    if args.ok_actions:
        fields['ok_actions'] = args.ok_actions
    if args.undetermined_actions:
        fields['undetermined_actions'] = args.undetermined_actions
    if args.severity:
        if not _validate_severity(args.severity):
            return
        fields['severity'] = args.severity
    if args.match_by:
        fields['match_by'] = args.match_by.split(',')
    try:
        alarm = mc.alarm_definitions.create(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        print(jsonutils.dumps(alarm, indent=2))
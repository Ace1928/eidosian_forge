import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('id', metavar='<ALARM_DEFINITION_ID>', help='The ID of the alarm definition.')
@utils.arg('name', metavar='<ALARM_DEFINITION_NAME>', help='Name of the alarm definition.')
@utils.arg('description', metavar='<DESCRIPTION>', help='Description of the alarm.')
@utils.arg('expression', metavar='<EXPRESSION>', help='The alarm expression to evaluate. Quoted.')
@utils.arg('alarm_actions', metavar='<ALARM-NOTIFICATION-ID1,ALARM-NOTIFICATION-ID2,...>', help='The notification method(s) to use when an alarm state is ALARM as a comma separated list.')
@utils.arg('ok_actions', metavar='<OK-NOTIFICATION-ID1,OK-NOTIFICATION-ID2,...>', help='The notification method(s) to use when an alarm state is OK as a comma separated list.')
@utils.arg('undetermined_actions', metavar='<UNDETERMINED-NOTIFICATION-ID1,UNDETERMINED-NOTIFICATION-ID2,...>', help='The notification method(s) to use when an alarm state is UNDETERMINED as a comma separated list.')
@utils.arg('actions_enabled', metavar='<ACTIONS-ENABLED>', help='The actions-enabled boolean is one of [true,false]')
@utils.arg('match_by', metavar='<MATCH_BY_DIMENSION_KEY1,MATCH_BY_DIMENSION_KEY2,...>', help='The metric dimensions to use to create unique alarms. One or more dimension key names separated by a comma. Key names need quoting when they contain special chars [&,(,),{,},>,<] that confuse the CLI parser.')
@utils.arg('severity', metavar='<SEVERITY>', help='Severity is one of [LOW, MEDIUM, HIGH, CRITICAL].')
def do_alarm_definition_update(mc, args):
    """Update the alarm definition."""
    fields = {}
    fields['alarm_id'] = args.id
    fields['name'] = args.name
    fields['description'] = args.description
    fields['expression'] = args.expression
    fields['alarm_actions'] = _arg_split_patch_update(args.alarm_actions)
    fields['ok_actions'] = _arg_split_patch_update(args.ok_actions)
    fields['undetermined_actions'] = _arg_split_patch_update(args.undetermined_actions)
    if args.actions_enabled not in enabled_types:
        errmsg = 'Invalid value, not one of [' + ', '.join(enabled_types) + ']'
        print(errmsg)
        return
    fields['actions_enabled'] = args.actions_enabled in ['true', 'True']
    fields['match_by'] = _arg_split_patch_update(args.match_by)
    if not _validate_severity(args.severity):
        return
    fields['severity'] = args.severity
    try:
        alarm = mc.alarm_definitions.update(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        print(jsonutils.dumps(alarm, indent=2))
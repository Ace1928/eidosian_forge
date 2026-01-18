import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('id', metavar='<ALARM_DEFINITION_ID>', help='The ID of the alarm definition.')
def do_alarm_definition_show(mc, args):
    """Describe the alarm definition."""
    fields = {}
    fields['alarm_id'] = args.id
    try:
        alarm = mc.alarm_definitions.get(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        if args.json:
            print(utils.json_formatter(alarm))
            return
        formatters = {'name': utils.json_formatter, 'id': utils.json_formatter, 'expression': utils.json_formatter, 'expression_data': utils.format_expression_data, 'match_by': utils.json_formatter, 'actions_enabled': utils.json_formatter, 'alarm_actions': utils.json_formatter, 'ok_actions': utils.json_formatter, 'severity': utils.json_formatter, 'undetermined_actions': utils.json_formatter, 'description': utils.json_formatter, 'links': utils.format_dictlist}
        utils.print_dict(alarm, formatters=formatters)
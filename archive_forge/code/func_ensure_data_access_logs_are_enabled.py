from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v2 import util as api_util
from googlecloudsdk.command_lib.projects import util as projects_util
def ensure_data_access_logs_are_enabled(trigger_event_filters):
    """Ensures appropriate Data Access Audit Logs are enabled for the given event filters.

  If they're not, the user will be prompted to enable them or warned if the
  console cannot prompt.

  Args:
    trigger_event_filters: the CAL trigger's event filters.
  """
    service_filter = [f for f in trigger_event_filters if f.attribute == 'serviceName']
    if service_filter:
        api_util.PromptToEnableDataAccessAuditLogs(service_filter[0].value)
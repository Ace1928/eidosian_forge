import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
def schedule_show(self, schedule, mistral_client=None):
    """Get details of a backup schedule.

        :param: schedule to show.
        :rtype: :class:`Schedule`.
        """
    if isinstance(schedule, Schedule):
        schedule = schedule.id
    if not mistral_client:
        mistral_client = self._get_mistral_client()
    schedule = mistral_client.cron_triggers.get(schedule)
    return self._build_schedule(schedule, schedule.workflow_input)
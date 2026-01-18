import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
def schedule_list(self, instance, mistral_client=None):
    """Get a list of all backup schedules for an instance.

        :param: instance for which to list schedules.
        :rtype: list of :class:`Schedule`.
        """
    inst_id = base.getid(instance)
    if not mistral_client:
        mistral_client = self._get_mistral_client()
    return [self._build_schedule(cron_trig, cron_trig.workflow_input) for cron_trig in mistral_client.cron_triggers.list() if inst_id in cron_trig.workflow_input]
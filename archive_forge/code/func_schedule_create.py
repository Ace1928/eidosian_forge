import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
def schedule_create(self, instance, pattern, name, description=None, incremental=None, mistral_client=None):
    """Create a new schedule to backup the given instance.

        :param instance: instance to backup.
        :param: pattern: cron pattern for schedule.
        :param name: name for backup.
        :param description: (optional).
        :param incremental: flag for incremental backup (optional).
        :returns: :class:`Backups`
        """
    if not mistral_client:
        mistral_client = self._get_mistral_client()
    inst_id = base.getid(instance)
    cron_name = str(uuid.uuid4())
    wf_input = {'instance': inst_id, 'name': name, 'description': description, 'incremental': incremental}
    cron_trigger = mistral_client.cron_triggers.create(cron_name, self.backup_create_workflow, pattern=pattern, workflow_input=wf_input)
    return self._build_schedule(cron_trigger, wf_input)
import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
def execution_list_generator():
    yielded = 0
    for sexec in mistral_execution_generator():
        if sexec.workflow_name == cron_trigger.workflow_name and ct_input == json.loads(sexec.input):
            yield ScheduleExecution(self, sexec.to_dict(), loaded=True)
            yielded += 1
        if limit and yielded == limit:
            return
import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_export_task(self, export_task_id: str=None, dry_run: bool=False):
    """
        Deletes an export task.
        If the export task is not running, the command fails and an error is
        returned.

        :param      export_task_id: The ID of the export task to delete.
        (required)
        :type       export_task_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteExportTask'
    data = {'DryRun': dry_run}
    if export_task_id is not None:
        data.update({'ExportTaskId': export_task_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()
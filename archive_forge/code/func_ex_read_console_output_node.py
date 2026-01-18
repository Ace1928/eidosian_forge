import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_read_console_output_node(self, node: Node, dry_run: bool=False):
    """
        Gets the console output for a virtual machine (VM). This console
        provides the most recent 64 KiB output.

        :param      node: the ID of the VM (required)
        :type       node: ``Node``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The Console Output of the specified Node.
        :rtype: ``str``
        """
    action = 'ReadConsoleOutput'
    data = json.dumps({'DryRun': dry_run, 'VmId': node.id})
    response = self._call_api(action, data)
    if response.status_code == 200:
        return response.json()['ConsoleOutput']
    return response.json()
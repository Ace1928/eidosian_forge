from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def delete_occm_agents(self, rest_api, agents):
    """
        delete a list of occm
        """
    results = []
    for agent in agents:
        if 'agentId' in agent:
            occm_status, error = self.delete_occm(rest_api, agent['agentId'])
        else:
            occm_status, error = (None, 'unexpected agent contents: %s' % repr(agent))
        if error:
            results.append((occm_status, error))
    return results
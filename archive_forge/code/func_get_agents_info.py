from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_agents_info(self, rest_api, headers):
    """
        Collect a list of agents matching account_id.
        :return: list of agents, error
        """
    account_id, error = self.get_account_id(rest_api)
    if error:
        return (None, error)
    agents, error = self.get_occm_agents_by_account(rest_api, account_id)
    return (agents, error)
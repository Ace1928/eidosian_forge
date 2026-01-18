from collections import namedtuple
import json
import logging
import pprint
import re
def filter_resp(self, action_resp, filter_params):
    """Filter response of action. Used to make printed results more
        specific

        :param action_resp: named tuple (CommandsResponse)
            containing response from action.
        :param filter_params: params used after '|' specific for given filter
        :return: filtered response.
        """
    if action_resp.status == STATUS_OK:
        try:
            return CommandsResponse(STATUS_OK, TextFilter.filter(action_resp.value, filter_params))
        except FilterError as e:
            return CommandsResponse(STATUS_ERROR, str(e))
    else:
        return action_resp
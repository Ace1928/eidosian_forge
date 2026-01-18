from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.plugins.callback import CallbackBase
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
def send_msg_v1(self, msg, msg_format='text', color='yellow', notify=False):
    """Method for sending a message to HipChat"""
    params = {}
    params['room_id'] = self.room
    params['from'] = self.from_name[:15]
    params['message'] = msg
    params['message_format'] = msg_format
    params['color'] = color
    params['notify'] = int(self.allow_notify and notify)
    url = '%s?auth_token=%s' % (self.API_V1_URL, self.token)
    try:
        response = open_url(url, data=urlencode(params))
        return response.read()
    except Exception as ex:
        self._display.warning('Could not submit message to hipchat: {0}'.format(ex))
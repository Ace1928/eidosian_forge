from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.plugins.callback import CallbackBase
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
def send_msg_v2(self, msg, msg_format='text', color='yellow', notify=False):
    """Method for sending a message to HipChat"""
    headers = {'Authorization': 'Bearer %s' % self.token, 'Content-Type': 'application/json'}
    body = {}
    body['room_id'] = self.room
    body['from'] = self.from_name[:15]
    body['message'] = msg
    body['message_format'] = msg_format
    body['color'] = color
    body['notify'] = self.allow_notify and notify
    data = json.dumps(body)
    url = self.API_V2_URL + 'room/{room_id}/notification'.format(room_id=self.room)
    try:
        response = open_url(url, data=data, headers=headers, method='POST')
        return response.read()
    except Exception as ex:
        self._display.warning('Could not submit message to hipchat: {0}'.format(ex))
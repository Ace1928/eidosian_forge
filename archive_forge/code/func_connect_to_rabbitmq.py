from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from mimetypes import MimeTypes
import os
import json
import traceback
def connect_to_rabbitmq(self):
    """
        Function to connect to rabbitmq using username and password
        """
    try:
        parameters = pika.URLParameters(self.url)
    except Exception as e:
        self.module.fail_json(msg='URL malformed: %s' % to_native(e))
    try:
        self.connection = pika.BlockingConnection(parameters)
    except Exception as e:
        self.module.fail_json(msg='Connection issue: %s' % to_native(e))
    try:
        self.conn_channel = self.connection.channel()
    except pika.exceptions.AMQPChannelError as e:
        self.close_connection()
        self.module.fail_json(msg='Channel issue: %s' % to_native(e))
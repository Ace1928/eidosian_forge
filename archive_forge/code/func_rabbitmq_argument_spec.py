from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from mimetypes import MimeTypes
import os
import json
import traceback
@staticmethod
def rabbitmq_argument_spec():
    return dict(url=dict(type='str'), proto=dict(type='str', choices=['amqp', 'amqps']), host=dict(type='str'), port=dict(type='int'), username=dict(type='str'), password=dict(type='str', no_log=True), vhost=dict(type='str'), queue=dict(type='str'))
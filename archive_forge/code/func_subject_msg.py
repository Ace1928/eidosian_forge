from __future__ import (absolute_import, division, print_function)
import json
import os
import re
import email.utils
import smtplib
from ansible.module_utils.common.text.converters import to_bytes
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.plugins.callback import CallbackBase
def subject_msg(self, multiline, failtype, linenr):
    return '%s: %s' % (failtype, multiline.strip('\r\n').splitlines()[linenr])
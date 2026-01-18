from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def answer_question(vm, responses):
    """Answer against the question for unlocking a virtual machine.

    Args:
        vm: Virtual machine management object
        responses: Answer contents to unlock a virtual machine
    """
    for response in responses:
        try:
            vm.AnswerVM(response['id'], response['response_num'])
        except Exception as e:
            raise TaskError('answer failed: %s' % to_text(e))
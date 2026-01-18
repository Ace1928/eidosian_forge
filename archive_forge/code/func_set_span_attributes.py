from __future__ import (absolute_import, division, print_function)
import getpass
import os
import socket
import sys
import time
import uuid
from collections import OrderedDict
from os.path import basename
from ansible.errors import AnsibleError
from ansible.module_utils.six import raise_from
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.plugins.callback import CallbackBase
def set_span_attributes(self, span, attributes):
    """ update the span attributes with the given attributes if not None """
    if span is None and self._display is not None:
        self._display.warning('span object is None. Please double check if that is expected.')
    elif attributes is not None:
        span.set_attributes(attributes)
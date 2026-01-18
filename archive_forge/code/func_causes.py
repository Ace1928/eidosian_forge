from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import exceptions
import six
@property
def causes(self):
    """Returns list of causes uniqued by the message."""
    try:
        messages = {c['message']: c for c in self._content['details']['causes']}
        return [messages[k] for k in sorted(messages)]
    except KeyError:
        return []
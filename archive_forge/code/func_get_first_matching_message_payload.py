from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import optimize_parameters_util
from googlecloudsdk.core import properties
def get_first_matching_message_payload(messages, topic):
    """Gets first item with matching topic from list of task output messages."""
    for message in messages:
        if topic is message.topic:
            return message.payload
    return None
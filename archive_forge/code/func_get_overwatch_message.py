from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def get_overwatch_message():
    client = get_overwatch_client()
    return client.MESSAGES_MODULE
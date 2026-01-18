from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import calendar
import datetime
import enum
import json
import textwrap
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core.resource import resource_projector
def convert_datetime_object_to_utc(datetime_object):
    """Converts datetime object to UTC and returns it."""
    offset = datetime_object.utcoffset()
    if offset:
        return (datetime_object - offset).replace(tzinfo=datetime.timezone.utc)
    return datetime_object
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
def get_metadata_json_section_string(key_string, value_to_convert_to_json):
    """Returns metadata section with potentially multiple lines of JSON.

  Args:
    key_string (str): Key to give section.
    value_to_convert_to_json (list|object): json_dump_method run on this.

  Returns:
    String with key followed by JSON version of value.
  """
    json_string = textwrap.indent(configured_json_dumps(value_to_convert_to_json), prefix=METADATA_LINE_INDENT_STRING)
    return '{indent}{key}:\n{json}'.format(indent=METADATA_LINE_INDENT_STRING, key=key_string, json=json_string)
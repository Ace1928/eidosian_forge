from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import shim_format_util
import six
def check_and_convert_to_readable_sizes(size, readable_sizes=False, use_gsutil_style=False):
    if readable_sizes and size is not None:
        return shim_format_util.get_human_readable_byte_value(size, use_gsutil_style=use_gsutil_style)
    else:
        return six.text_type(size)
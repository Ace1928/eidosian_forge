from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import datetime
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
import six
def get_formatted_string(resource, display_titles_and_defaults, show_acl=True, show_version_in_url=False):
    """Returns the formatted string representing the resource.

  Args:
    resource (resource_reference.Resource): Object holding resource metadata
      that needs to be displayed.
    display_titles_and_defaults ([Bucket|Object]DisplayTitlesAndDefaults): Holds
      the display titles and default values for each field present in the
      Resource.
    show_acl (bool): Include ACLs list in resource display.
    show_version_in_url (bool): Display extended URL with versioning info.

  Returns:
    A string representing the Resource for ls -L command.
  """
    lines = []
    if show_acl:
        formatted_acl_dict = resource.get_formatted_acl()
    else:
        formatted_acl_dict = {}
    for key in display_titles_and_defaults._fields:
        if not show_acl and key == ACL_KEY:
            continue
        field_display_title_and_default = getattr(display_titles_and_defaults, key)
        if field_display_title_and_default is None:
            continue
        if field_display_title_and_default.field_name is not None:
            field_name = field_display_title_and_default.field_name
        else:
            field_name = key
        if field_name in formatted_acl_dict:
            value = formatted_acl_dict.get(field_name)
        else:
            value = getattr(resource, field_name, None)
        if value == resource_reference.NOT_SUPPORTED_DO_NOT_DISPLAY:
            continue
        line = _get_formatted_line(field_display_title_and_default.title, value, field_display_title_and_default.default)
        if line:
            lines.append(line)
    if show_version_in_url:
        url_string = resource.storage_url.url_string
    else:
        url_string = resource.storage_url.versionless_url_string
    return '{url_string}:\n{fields}'.format(url_string=url_string, fields='\n'.join(lines))
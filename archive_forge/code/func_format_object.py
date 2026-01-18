from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter as base
from googlecloudsdk.command_lib.storage.resources import shim_format_util
def format_object(self, object_resource, show_acl=True, show_version_in_url=False, **kwargs):
    """See super class."""
    del kwargs
    shim_format_util.replace_object_values_with_encryption_string(object_resource, 'Underlying data encrypted')
    return base.get_formatted_string(object_resource, _OBJECT_DISPLAY_TITLES_AND_DEFAULTS, show_acl=show_acl, show_version_in_url=show_version_in_url)
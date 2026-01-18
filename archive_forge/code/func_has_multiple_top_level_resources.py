from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
@property
def has_multiple_top_level_resources(self):
    """Returns if the iterator yields plural items without recursing.

    Also returns True if the iterator was created with multiple URLs.
    This may not be true if one URL doesn't return anything, but it's
    consistent with gsutil and the user's probable intentions.

    Returns:
      Boolean indicating if iterator contains multiple top-level sources.
    """
    if self._has_multiple_top_level_resources is None:
        self._has_multiple_top_level_resources = self._urls_iterator.is_plural() or self._top_level_iterator.is_plural()
    return self._has_multiple_top_level_resources
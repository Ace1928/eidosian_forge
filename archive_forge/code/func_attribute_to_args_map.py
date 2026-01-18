from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import info_holders
@property
def attribute_to_args_map(self):
    """The map of attribute names to associated args.

    Returns:
      {str: str}, the map.
    """
    return self._attribute_to_args_map
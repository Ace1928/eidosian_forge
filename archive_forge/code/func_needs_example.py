from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def needs_example(self):
    """Check whether command requires an example."""
    if self.command_metadata and self.command_metadata.is_group:
        return False
    if 'alpha' in self.command_name:
        return False
    for name in self._NON_COMMAND_SURFACE_GROUPS:
        if self.command_name.startswith(name):
            return False
    return True
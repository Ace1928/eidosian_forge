from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import shlex
def get_argument_metadatas(self):
    """Returns the metadata for an entire example command string."""
    return sorted(self._argument_metadatas, key=lambda x: x.stop_index)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.cache import cache_update_ops
@property
def cache_updater(self):
    return self._cache_updater
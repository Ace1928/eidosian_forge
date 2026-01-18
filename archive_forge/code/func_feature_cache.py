from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import command
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
def feature_cache(self, refresh: bool=False):
    """Gets and caches the current feature for this object."""
    cache = getattr(self, '__feature_cache', None)
    if cache is None or refresh:
        cache = self.GetFeature()
        setattr(self, '__feature_cache', cache)
    return cache
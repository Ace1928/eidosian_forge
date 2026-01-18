from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def app_to_project_id(self, app_id):
    """Converts a string app id to a string project id."""
    return self._id_resolver.resolve_project_id(app_id)
from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_client_id(self):
    if self.has_client_id_:
        self.has_client_id_ = 0
        self.client_id_ = ''
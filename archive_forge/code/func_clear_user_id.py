from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_user_id(self):
    if self.has_user_id_:
        self.has_user_id_ = 0
        self.user_id_ = ''
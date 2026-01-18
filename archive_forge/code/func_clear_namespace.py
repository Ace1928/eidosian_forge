from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_namespace(self):
    if self.has_namespace_:
        self.has_namespace_ = 0
        self.namespace_ = ''
from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def mutable_default_config(self):
    self.has_default_config_ = 1
    return self.default_config()
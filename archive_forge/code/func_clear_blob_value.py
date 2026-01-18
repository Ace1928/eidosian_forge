from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_blob_value(self):
    if self.has_blob_value_:
        self.has_blob_value_ = 0
        self.blob_value_ = ''
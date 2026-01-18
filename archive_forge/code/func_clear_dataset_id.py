from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_dataset_id(self):
    if self.has_dataset_id_:
        self.has_dataset_id_ = 0
        self.dataset_id_ = ''
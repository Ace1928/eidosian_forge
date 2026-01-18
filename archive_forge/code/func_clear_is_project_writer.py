from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_is_project_writer(self):
    if self.has_is_project_writer_:
        self.has_is_project_writer_ = 0
        self.is_project_writer_ = 0
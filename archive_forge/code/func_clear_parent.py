from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_parent(self):
    if self.has_parent_:
        self.has_parent_ = 0
        self.parent_ = 0
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_disabled_index(self):
    if self.has_disabled_index_:
        self.has_disabled_index_ = 0
        self.disabled_index_ = 0
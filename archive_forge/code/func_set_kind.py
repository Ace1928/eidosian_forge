from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_kind(self, x):
    self.has_kind_ = 1
    self.kind_ = x
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def mutable_value(self):
    self.has_value_ = 1
    return self.value_
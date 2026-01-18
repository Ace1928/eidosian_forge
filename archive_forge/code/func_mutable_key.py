from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def mutable_key(self):
    self.has_key_ = 1
    return self.key()
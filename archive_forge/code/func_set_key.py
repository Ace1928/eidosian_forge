from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_key(self, x):
    self.has_key_ = 1
    self.key_ = x
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_before(self, x):
    self.has_before_ = 1
    self.before_ = x
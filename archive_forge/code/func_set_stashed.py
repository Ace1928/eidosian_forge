from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_stashed(self, x):
    self.has_stashed_ = 1
    self.stashed_ = x
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_only_use_if_required(self, x):
    self.has_only_use_if_required_ = 1
    self.only_use_if_required_ = x
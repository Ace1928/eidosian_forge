from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_only_use_if_required(self):
    if self.has_only_use_if_required_:
        self.has_only_use_if_required_ = 0
        self.only_use_if_required_ = 0
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_stashed(self):
    if self.has_stashed_:
        self.has_stashed_ = 0
        self.stashed_ = -1
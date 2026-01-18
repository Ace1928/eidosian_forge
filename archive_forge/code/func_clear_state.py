from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_state(self):
    if self.has_state_:
        self.has_state_ = 0
        self.state_ = 0
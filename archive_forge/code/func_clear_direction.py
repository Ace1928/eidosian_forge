from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_direction(self):
    if self.has_direction_:
        self.has_direction_ = 0
        self.direction_ = 0
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_int64value(self):
    if self.has_int64value_:
        self.has_int64value_ = 0
        self.int64value_ = 0
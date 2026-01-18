from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_before_ascending(self):
    if self.has_before_ascending_:
        self.has_before_ascending_ = 0
        self.before_ascending_ = 0
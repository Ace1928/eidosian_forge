from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_uservalue(self):
    if self.has_uservalue_:
        self.has_uservalue_ = 0
        if self.uservalue_ is not None:
            self.uservalue_.Clear()
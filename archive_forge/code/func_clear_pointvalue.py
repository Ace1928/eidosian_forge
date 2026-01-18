from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_pointvalue(self):
    if self.has_pointvalue_:
        self.has_pointvalue_ = 0
        if self.pointvalue_ is not None:
            self.pointvalue_.Clear()
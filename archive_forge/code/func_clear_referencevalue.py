from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_referencevalue(self):
    if self.has_referencevalue_:
        self.has_referencevalue_ = 0
        if self.referencevalue_ is not None:
            self.referencevalue_.Clear()
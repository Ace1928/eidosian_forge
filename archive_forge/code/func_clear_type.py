from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_type(self):
    if self.has_type_:
        self.has_type_ = 0
        self.type_ = ''
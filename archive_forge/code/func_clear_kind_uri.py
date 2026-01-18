from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_kind_uri(self):
    if self.has_kind_uri_:
        self.has_kind_uri_ = 0
        self.kind_uri_ = ''
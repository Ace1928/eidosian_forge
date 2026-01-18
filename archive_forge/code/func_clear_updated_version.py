from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_updated_version(self):
    if self.has_updated_version_:
        self.has_updated_version_ = 0
        self.updated_version_ = 0
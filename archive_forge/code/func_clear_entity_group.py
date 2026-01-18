from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_entity_group(self):
    self.has_entity_group_ = 0
    self.entity_group_.Clear()
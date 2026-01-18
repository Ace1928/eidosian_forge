from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_property_type_for_stats(self):
    if self.has_property_type_for_stats_:
        self.has_property_type_for_stats_ = 0
        self.property_type_for_stats_ = ''
from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def mutable_geo_point_value(self):
    self.has_geo_point_value_ = 1
    return self.geo_point_value()
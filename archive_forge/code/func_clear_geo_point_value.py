from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_geo_point_value(self):
    if self.has_geo_point_value_:
        self.has_geo_point_value_ = 0
        if self.geo_point_value_ is not None:
            self.geo_point_value_.Clear()
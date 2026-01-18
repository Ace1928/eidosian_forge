from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_default_config(self):
    if self.has_default_config_:
        self.has_default_config_ = 0
        if self.default_config_ is not None:
            self.default_config_.Clear()
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_ts(self):
    if self.has_ts_:
        self.has_ts_ = 0
        self.ts_ = 0
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_error_message(self):
    if self.has_error_message_:
        self.has_error_message_ = 0
        self.error_message_ = ''
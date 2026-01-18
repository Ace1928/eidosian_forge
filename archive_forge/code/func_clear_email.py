from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_email(self):
    if self.has_email_:
        self.has_email_ = 0
        self.email_ = ''
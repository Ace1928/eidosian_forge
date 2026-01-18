from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_login_url(self):
    if self.has_login_url_:
        self.has_login_url_ = 0
        self.login_url_ = ''
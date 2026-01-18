from .message_text import TextMessageProtocolEntity
from yowsup.structs import ProtocolTreeNode
import time
def setBroadcastProps(self, jids):
    assert type(jids) is list, 'jids must be a list, got %s instead.' % type(jids)
    self.jids = jids
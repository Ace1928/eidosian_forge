import codecs
import errno
import os
import random
import sys
import ovs.json
import ovs.poller
import ovs.reconnect
import ovs.stream
import ovs.timeval
import ovs.util
import ovs.vlog
@staticmethod
def create_notify(method, params):
    return Message(Message.T_NOTIFY, method, params, None, None, None)
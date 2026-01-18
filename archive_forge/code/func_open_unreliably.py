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
def open_unreliably(jsonrpc):
    reconnect = ovs.reconnect.Reconnect(ovs.timeval.msec())
    session = Session(reconnect, None, [jsonrpc.name])
    reconnect.set_quiet(True)
    session.pick_remote()
    reconnect.set_max_tries(0)
    reconnect.connected(ovs.timeval.msec())
    return session
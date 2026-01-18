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
def pick_remote(self):
    self.reconnect.set_name(self.remotes[self.next_remote])
    self.next_remote = (self.next_remote + 1) % len(self.remotes)
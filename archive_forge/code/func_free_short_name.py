import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def free_short_name(short_name):
    if short_name is None:
        return
    link_name = os.path.dirname(short_name)
    ovs.fatal_signal.unlink_file_now(link_name)
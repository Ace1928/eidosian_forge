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
def open_multiple(remotes, probe_interval=None):
    reconnect = ovs.reconnect.Reconnect(ovs.timeval.msec())
    session = Session(reconnect, None, remotes)
    session.pick_remote()
    reconnect.enable(ovs.timeval.msec())
    reconnect.set_backoff_free_tries(len(remotes))
    if ovs.stream.PassiveStream.is_valid_name(reconnect.get_name()):
        reconnect.set_passive(True, ovs.timeval.msec())
    if not ovs.stream.stream_or_pstream_needs_probes(reconnect.get_name()):
        reconnect.set_probe_interval(0)
    elif probe_interval is not None:
        reconnect.set_probe_interval(probe_interval)
    return session
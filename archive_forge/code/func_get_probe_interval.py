import os
import ovs.util
import ovs.vlog
def get_probe_interval(self):
    """Returns the "probe interval" in milliseconds.  If this is zero, it
        disables the connection keepalive feature.  If it is nonzero, then if
        the interval passes while the FSM is connected and without
        self.activity() being called, self.run() returns ovs.reconnect.PROBE.
        If the interval passes again without self.activity() being called,
        self.run() returns ovs.reconnect.DISCONNECT."""
    return self.probe_interval
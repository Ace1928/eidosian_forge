import os
import ovs.util
import ovs.vlog
def connect_failed(self, now, error):
    """Tell this FSM that the connection attempt failed.

        The FSM will back off and attempt to reconnect."""
    self.connecting(now)
    self.disconnected(now, error)
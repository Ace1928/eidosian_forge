import threading, inspect, shlex
def queueCmd(self, cmd):
    self._queuedCmds.append(cmd)
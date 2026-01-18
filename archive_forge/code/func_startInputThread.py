import threading, inspect, shlex
def startInputThread(self):
    global input
    try:
        input = raw_input
    except NameError:
        pass
    while True:
        cmd = self._queuedCmds.pop(0) if len(self._queuedCmds) else input(self.getPrompt()).strip()
        wait = self.execCmd(cmd)
        if wait:
            self.acceptingInput = False
            self.blockingQueue.get(True)
        self.acceptingInput = True
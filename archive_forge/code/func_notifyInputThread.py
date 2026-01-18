import threading, inspect, shlex
def notifyInputThread(self):
    self.blockingQueue.put(1)
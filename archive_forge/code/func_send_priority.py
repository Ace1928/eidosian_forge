import collections
def send_priority(self, name, data=None):
    self._queue.appendleft(Message(name, data))
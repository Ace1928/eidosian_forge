from collections import deque
import select
import msgpack
def peek_notification(self):
    while True:
        rlist, _wlist = self._endpoint.selectable()
        rlist, _wlist, _xlist = select.select(rlist, [], [], 0)
        if not rlist:
            break
        self.receive_notification()
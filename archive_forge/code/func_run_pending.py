import datetime
import numbers
import abc
import bisect
import pytz
def run_pending(self):
    while self.queue:
        command = self.queue[0]
        if not command.due():
            break
        self.run(command)
        if isinstance(command, PeriodicCommand):
            self.add(command.next())
        del self.queue[0]
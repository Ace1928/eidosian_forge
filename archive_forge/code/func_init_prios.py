import hashlib
import logging
from scrapy.utils.misc import create_instance
def init_prios(self, startprios):
    if not startprios:
        return
    for priority in startprios:
        self.queues[priority] = self.qfactory(priority)
    self.curprio = min(startprios)
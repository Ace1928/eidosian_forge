import logging
import socket
import sys
from importlib import import_module
from pprint import pformat
from twisted.internet import task
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.mail import MailSender
from scrapy.utils.engine import get_engine_status
def engine_started(self):
    self.crawler.stats.set_value('memusage/startup', self.get_virtual_size())
    self.tasks = []
    tsk = task.LoopingCall(self.update)
    self.tasks.append(tsk)
    tsk.start(self.check_interval, now=True)
    if self.limit:
        tsk = task.LoopingCall(self._check_limit)
        self.tasks.append(tsk)
        tsk.start(self.check_interval, now=True)
    if self.warning:
        tsk = task.LoopingCall(self._check_warning)
        self.tasks.append(tsk)
        tsk.start(self.check_interval, now=True)
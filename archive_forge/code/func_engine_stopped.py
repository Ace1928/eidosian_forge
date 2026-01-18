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
def engine_stopped(self):
    for tsk in self.tasks:
        if tsk.running:
            tsk.stop()
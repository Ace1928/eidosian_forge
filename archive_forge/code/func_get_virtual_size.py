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
def get_virtual_size(self):
    size = self.resource.getrusage(self.resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        size *= 1024
    return size
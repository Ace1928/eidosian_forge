import binascii
import logging
import os
import pprint
import traceback
from twisted.internet import protocol
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.utils.decorators import defers
from scrapy.utils.engine import print_engine_status
from scrapy.utils.reactor import listen_tcp
from scrapy.utils.trackref import print_live_refs
def start_listening(self):
    self.port = listen_tcp(self.portrange, self.host, self)
    h = self.port.getHost()
    logger.info('Telnet console listening on %(host)s:%(port)d', {'host': h.host, 'port': h.port}, extra={'crawler': self.crawler})
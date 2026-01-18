import logging
import signal
import sys
import threading
import traceback
from pdb import Pdb
from scrapy.utils.engine import format_engine_status
from scrapy.utils.trackref import format_live_refs
def dump_stacktrace(self, signum, frame):
    log_args = {'stackdumps': self._thread_stacks(), 'enginestatus': format_engine_status(self.crawler.engine), 'liverefs': format_live_refs()}
    logger.info('Dumping stack trace and engine status\n%(enginestatus)s\n%(liverefs)s\n%(stackdumps)s', log_args, extra={'crawler': self.crawler})
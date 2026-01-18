import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from twisted.python import failure
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import UsageError
from scrapy.utils.conf import arglist_to_dict, feed_process_params_from_cli
def set_crawler(self, crawler):
    if hasattr(self, '_crawler'):
        raise RuntimeError('crawler already set')
    self._crawler = crawler
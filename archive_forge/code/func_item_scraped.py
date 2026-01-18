import logging
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse
from twisted.internet import defer, threads
from twisted.internet.defer import DeferredList
from w3lib.url import file_uri_to_path
from zope.interface import Interface, implementer
from scrapy import Spider, signals
from scrapy.exceptions import NotConfigured, ScrapyDeprecationWarning
from scrapy.extensions.postprocessing import PostProcessingManager
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.conf import feed_complete_default_values_from_settings
from scrapy.utils.defer import maybe_deferred_to_future
from scrapy.utils.deprecate import create_deprecated_class
from scrapy.utils.ftp import ftp_store_file
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.python import get_func_args, without_none_values
def item_scraped(self, item, spider):
    slots = []
    for slot in self.slots:
        if not slot.filter.accepts(item):
            slots.append(slot)
            continue
        slot.start_exporting()
        slot.exporter.export_item(item)
        slot.itemcount += 1
        if self.feeds[slot.uri_template]['batch_item_count'] and slot.itemcount >= self.feeds[slot.uri_template]['batch_item_count']:
            uri_params = self._get_uri_params(spider, self.feeds[slot.uri_template]['uri_params'], slot)
            self._close_slot(slot, spider)
            slots.append(self._start_new_batch(batch_id=slot.batch_id + 1, uri=slot.uri_template % uri_params, feed_options=self.feeds[slot.uri_template], spider=spider, uri_template=slot.uri_template))
        else:
            slots.append(slot)
    self.slots = slots
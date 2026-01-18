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
def start_exporting(self):
    if not self._fileloaded:
        self.file = self.storage.open(self.spider)
        if 'postprocessing' in self.feed_options:
            self.file = PostProcessingManager(self.feed_options['postprocessing'], self.file, self.feed_options)
        self.exporter = self._get_exporter(file=self.file, format=self.feed_options['format'], fields_to_export=self.feed_options['fields'], encoding=self.feed_options['encoding'], indent=self.feed_options['indent'], **self.feed_options['item_export_kwargs'])
        self._fileloaded = True
    if not self._exporting:
        self.exporter.start_exporting()
        self._exporting = True
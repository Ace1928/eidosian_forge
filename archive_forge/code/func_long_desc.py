import sys
from argparse import Namespace
from typing import List, Type
from w3lib.url import is_url
from scrapy import Spider
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.http import Request
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.spider import DefaultSpider, spidercls_for_request
def long_desc(self):
    return 'Fetch a URL using the Scrapy downloader and print its content to stdout. You may want to use --nolog to disable logging'
from scrapy.exceptions import NotConfigured, NotSupported
from scrapy.selector import Selector
from scrapy.spiders import Spider
from scrapy.utils.iterators import csviter, xmliter_lxml
from scrapy.utils.spider import iterate_spider_output
def parse_row(self, response, row):
    """This method must be overridden with your custom spider functionality"""
    raise NotImplementedError
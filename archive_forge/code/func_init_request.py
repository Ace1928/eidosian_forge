from scrapy.spiders import Spider
from scrapy.utils.spider import iterate_spider_output
def init_request(self):
    """This function should return one initialization request, with the
        self.initialized method as callback. When the self.initialized method
        is called this spider is considered initialized. If you need to perform
        several requests for initializing your spider, you can do so by using
        different callbacks. The only requirement is that the final callback
        (of the last initialization request) must be self.initialized.

        The default implementation calls self.initialized immediately, and
        means that no initialization is needed. This method should be
        overridden only when you need to perform requests to initialize your
        spider
        """
    return self.initialized()
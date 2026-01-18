import logging
from twisted.internet.defer import Deferred, maybeDeferred
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import load_object
def robot_parser(self, request, spider):
    url = urlparse_cached(request)
    netloc = url.netloc
    if netloc not in self._parsers:
        self._parsers[netloc] = Deferred()
        robotsurl = f'{url.scheme}://{url.netloc}/robots.txt'
        robotsreq = Request(robotsurl, priority=self.DOWNLOAD_PRIORITY, meta={'dont_obey_robotstxt': True}, callback=NO_CALLBACK)
        dfd = self.crawler.engine.download(robotsreq)
        dfd.addCallback(self._parse_robots, netloc, spider)
        dfd.addErrback(self._logerror, robotsreq, spider)
        dfd.addErrback(self._robots_error, netloc)
        self.crawler.stats.inc_value('robotstxt/request_count')
    if isinstance(self._parsers[netloc], Deferred):
        d = Deferred()

        def cb(result):
            d.callback(result)
            return result
        self._parsers[netloc].addCallback(cb)
        return d
    return self._parsers[netloc]
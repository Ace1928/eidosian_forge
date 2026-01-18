from __future__ import annotations
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, cast
from twisted.internet.defer import Deferred
from scrapy.crawler import Crawler
from scrapy.dupefilters import BaseDupeFilter
from scrapy.http.request import Request
from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector
from scrapy.utils.job import job_dir
from scrapy.utils.misc import create_instance, load_object
def next_request(self) -> Optional[Request]:
    """
        Return a :class:`~scrapy.http.Request` object from the memory queue,
        falling back to the disk queue if the memory queue is empty.
        Return ``None`` if there are no more enqueued requests.

        Increment the appropriate stats, such as: ``scheduler/dequeued``,
        ``scheduler/dequeued/disk``, ``scheduler/dequeued/memory``.
        """
    request: Optional[Request] = self.mqs.pop()
    assert self.stats is not None
    if request is not None:
        self.stats.inc_value('scheduler/dequeued/memory', spider=self.spider)
    else:
        request = self._dqpop()
        if request is not None:
            self.stats.inc_value('scheduler/dequeued/disk', spider=self.spider)
    if request is not None:
        self.stats.inc_value('scheduler/dequeued', spider=self.spider)
    return request
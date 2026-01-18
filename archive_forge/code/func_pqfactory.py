import hashlib
import logging
from scrapy.utils.misc import create_instance
def pqfactory(self, slot, startprios=()):
    return ScrapyPriorityQueue(self.crawler, self.downstream_queue_cls, self.key + '/' + _path_safe(slot), startprios)
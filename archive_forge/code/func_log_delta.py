import logging
from datetime import datetime, timezone
from twisted.internet import task
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.utils.serialize import ScrapyJSONEncoder
def log_delta(self):
    num_stats = {k: v for k, v in self.stats._stats.items() if isinstance(v, (int, float)) and self.param_allowed(k, self.ext_delta_include, self.ext_delta_exclude)}
    delta = {k: v - self.delta_prev.get(k, 0) for k, v in num_stats.items()}
    self.delta_prev = num_stats
    return {'delta': delta}
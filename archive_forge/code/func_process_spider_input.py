import logging
from scrapy.exceptions import IgnoreRequest
def process_spider_input(self, response, spider):
    if 200 <= response.status < 300:
        return
    meta = response.meta
    if meta.get('handle_httpstatus_all', False):
        return
    if 'handle_httpstatus_list' in meta:
        allowed_statuses = meta['handle_httpstatus_list']
    elif self.handle_httpstatus_all:
        return
    else:
        allowed_statuses = getattr(spider, 'handle_httpstatus_list', self.handle_httpstatus_list)
    if response.status in allowed_statuses:
        return
    raise HttpError(response, 'Ignoring non-200 response')
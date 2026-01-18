import hashlib
import logging
from scrapy.utils.misc import create_instance
def get_slot_key(self, request):
    return self.downloader._get_slot_key(request, None)
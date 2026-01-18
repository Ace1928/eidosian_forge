import functools
import hashlib
import warnings
from contextlib import suppress
from io import BytesIO
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem, NotConfigured, ScrapyDeprecationWarning
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.pipelines.files import FileException, FilesPipeline
from scrapy.settings import Settings
from scrapy.utils.misc import md5sum
from scrapy.utils.python import get_func_args, to_bytes
def image_downloaded(self, response, request, info, *, item=None):
    checksum = None
    for path, image, buf in self.get_images(response, request, info, item=item):
        if checksum is None:
            buf.seek(0)
            checksum = md5sum(buf)
        width, height = image.size
        self.store.persist_file(path, buf, info, meta={'width': width, 'height': height}, headers={'Content-Type': 'image/jpeg'})
    return checksum
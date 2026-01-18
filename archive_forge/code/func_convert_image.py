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
def convert_image(self, image, size=None, response_body=None):
    if response_body is None:
        warnings.warn(f'{self.__class__.__name__}.convert_image() method called in a deprecated way, method called without response_body argument.', category=ScrapyDeprecationWarning, stacklevel=2)
    if image.format in ('PNG', 'WEBP') and image.mode == 'RGBA':
        background = self._Image.new('RGBA', image.size, (255, 255, 255))
        background.paste(image, image)
        image = background.convert('RGB')
    elif image.mode == 'P':
        image = image.convert('RGBA')
        background = self._Image.new('RGBA', image.size, (255, 255, 255))
        background.paste(image, image)
        image = background.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    if size:
        image = image.copy()
        try:
            resampling_filter = self._Image.Resampling.LANCZOS
        except AttributeError:
            resampling_filter = self._Image.ANTIALIAS
        image.thumbnail(size, resampling_filter)
    elif response_body is not None and image.format == 'JPEG':
        return (image, response_body)
    buf = BytesIO()
    image.save(buf, 'JPEG')
    return (image, buf)
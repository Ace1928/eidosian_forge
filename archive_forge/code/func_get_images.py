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
def get_images(self, response, request, info, *, item=None):
    path = self.file_path(request, response=response, info=info, item=item)
    orig_image = self._Image.open(BytesIO(response.body))
    width, height = orig_image.size
    if width < self.min_width or height < self.min_height:
        raise ImageException(f'Image too small ({width}x{height} < {self.min_width}x{self.min_height})')
    if self._deprecated_convert_image is None:
        self._deprecated_convert_image = 'response_body' not in get_func_args(self.convert_image)
        if self._deprecated_convert_image:
            warnings.warn(f'{self.__class__.__name__}.convert_image() method overridden in a deprecated way, overridden method does not accept response_body argument.', category=ScrapyDeprecationWarning)
    if self._deprecated_convert_image:
        image, buf = self.convert_image(orig_image)
    else:
        image, buf = self.convert_image(orig_image, response_body=BytesIO(response.body))
    yield (path, image, buf)
    for thumb_id, size in self.thumbs.items():
        thumb_path = self.thumb_path(request, thumb_id, response=response, info=info, item=item)
        if self._deprecated_convert_image:
            thumb_image, thumb_buf = self.convert_image(image, size)
        else:
            thumb_image, thumb_buf = self.convert_image(image, size, buf)
        yield (thumb_path, thumb_image, thumb_buf)
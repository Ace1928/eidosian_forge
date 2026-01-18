from __future__ import annotations
import io
import os
import re
from enum import IntEnum
from typing import TYPE_CHECKING, Final, List, Literal, Sequence, Union, cast
from typing_extensions import TypeAlias
from streamlit import runtime, url_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Image_pb2 import ImageList as ImageListProto
from streamlit.runtime import caching
from streamlit.runtime.metrics_util import gather_metrics
def image_to_url(image: AtomicImage, width: int, clamp: bool, channels: Channels, output_format: ImageFormatOrAuto, image_id: str) -> str:
    """Return a URL that an image can be served from.
    If `image` is already a URL, return it unmodified.
    Otherwise, add the image to the MediaFileManager and return the URL.

    (When running in "raw" mode, we won't actually load data into the
    MediaFileManager, and we'll return an empty URL.)
    """
    import numpy as np
    from PIL import Image, ImageFile
    image_data: bytes
    if isinstance(image, str):
        if not os.path.isfile(image) and url_util.is_url(image, allowed_schemas=('http', 'https', 'data')):
            return image
        if image.endswith('.svg') and os.path.isfile(image):
            with open(image) as textfile:
                image = textfile.read()
        if re.search('(^\\s?(<\\?xml[\\s\\S]*<svg\\s)|^\\s?<svg\\s|^\\s?<svg>\\s)', image):
            if 'xmlns' not in image:
                image = image.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
            import base64
            image_b64_encoded = base64.b64encode(image.encode('utf-8')).decode('utf-8')
            return f'data:image/svg+xml;base64,{image_b64_encoded}'
        try:
            with open(image, 'rb') as f:
                image_data = f.read()
        except Exception:
            import mimetypes
            mimetype, _ = mimetypes.guess_type(image)
            if mimetype is None:
                mimetype = 'application/octet-stream'
            url = runtime.get_instance().media_file_mgr.add(image, mimetype, image_id)
            caching.save_media_data(image, mimetype, image_id)
            return url
    elif isinstance(image, (ImageFile.ImageFile, Image.Image)):
        format = _validate_image_format_string(image, output_format)
        image_data = _PIL_to_bytes(image, format)
    elif isinstance(image, io.BytesIO):
        image_data = _BytesIO_to_bytes(image)
    elif isinstance(image, np.ndarray):
        image = _clip_image(_verify_np_shape(image), clamp)
        if channels == 'BGR':
            if len(image.shape) == 3:
                image = image[:, :, [2, 1, 0]]
            else:
                raise StreamlitAPIException('When using `channels="BGR"`, the input image should have exactly 3 color channels')
        image_data = _np_array_to_bytes(array=cast('npt.NDArray[Any]', image), output_format=output_format)
    else:
        image_data = image
    image_format = _validate_image_format_string(image_data, output_format)
    image_data = _ensure_image_size_and_format(image_data, width, image_format)
    mimetype = _get_image_format_mimetype(image_format)
    if runtime.exists():
        url = runtime.get_instance().media_file_mgr.add(image_data, mimetype, image_id)
        caching.save_media_data(image_data, mimetype, image_id)
        return url
    else:
        return ''
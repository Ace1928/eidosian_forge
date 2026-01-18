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
def marshall_images(coordinates: str, image: ImageOrImageList, caption: str | npt.NDArray[Any] | list[str] | None, width: int | WidthBehaviour, proto_imgs: ImageListProto, clamp: bool, channels: Channels='RGB', output_format: ImageFormatOrAuto='auto') -> None:
    """Fill an ImageListProto with a list of images and their captions.

    The images will be resized and reformatted as necessary.

    Parameters
    ----------
    coordinates
        A string indentifying the images' location in the frontend.
    image
        The image or images to include in the ImageListProto.
    caption
        Image caption. If displaying multiple images, caption should be a
        list of captions (one for each image).
    width
        The desired width of the image or images. This parameter will be
        passed to the frontend.
        Positive values set the image width explicitly.
        Negative values has some special. For details, see: `WidthBehaviour`
    proto_imgs
        The ImageListProto to fill in.
    clamp
        Clamp image pixel values to a valid range ([0-255] per channel).
        This is only meaningful for byte array images; the parameter is
        ignored for image URLs. If this is not set, and an image has an
        out-of-range value, an error will be thrown.
    channels
        If image is an nd.array, this parameter denotes the format used to
        represent color information. Defaults to 'RGB', meaning
        `image[:, :, 0]` is the red channel, `image[:, :, 1]` is green, and
        `image[:, :, 2]` is blue. For images coming from libraries like
        OpenCV you should set this to 'BGR', instead.
    output_format
        This parameter specifies the format to use when transferring the
        image data. Photos should use the JPEG format for lossy compression
        while diagrams should use the PNG format for lossless compression.
        Defaults to 'auto' which identifies the compression type based
        on the type and format of the image argument.
    """
    import numpy as np
    channels = cast(Channels, channels.upper())
    images: Sequence[AtomicImage]
    if isinstance(image, list):
        images = image
    elif isinstance(image, np.ndarray) and len(image.shape) == 4:
        images = _4d_to_list_3d(image)
    else:
        images = [image]
    if type(caption) is list:
        captions: Sequence[str | None] = caption
    elif isinstance(caption, str):
        captions = [caption]
    elif isinstance(caption, np.ndarray) and len(caption.shape) == 1:
        captions = caption.tolist()
    elif caption is None:
        captions = [None] * len(images)
    else:
        captions = [str(caption)]
    assert type(captions) == list, 'If image is a list then caption should be as well'
    assert len(captions) == len(images), 'Cannot pair %d captions with %d images.' % (len(captions), len(images))
    proto_imgs.width = int(width)
    for coord_suffix, (image, caption) in enumerate(zip(images, captions)):
        proto_img = proto_imgs.imgs.add()
        if caption is not None:
            proto_img.caption = str(caption)
        image_id = '%s-%i' % (coordinates, coord_suffix)
        proto_img.url = image_to_url(image, width, clamp, channels, output_format, image_id)
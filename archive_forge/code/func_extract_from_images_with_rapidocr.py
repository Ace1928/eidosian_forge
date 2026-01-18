from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
def extract_from_images_with_rapidocr(images: Sequence[Union[Iterable[np.ndarray], bytes]]) -> str:
    """Extract text from images with RapidOCR.

    Args:
        images: Images to extract text from.

    Returns:
        Text extracted from images.

    Raises:
        ImportError: If `rapidocr-onnxruntime` package is not installed.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        raise ImportError('`rapidocr-onnxruntime` package not found, please install it with `pip install rapidocr-onnxruntime`')
    ocr = RapidOCR()
    text = ''
    for img in images:
        result, _ = ocr(img)
        if result:
            result = [text[1] for text in result]
            text += '\n'.join(result)
    return text
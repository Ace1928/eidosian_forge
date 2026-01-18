import unittest
import io
import os
import asyncio
from PIL import Image, ImageEnhance, ExifTags
import hashlib
import zstd
from typing import Tuple, Dict, Any, Optional, List, Callable
from core_services import (
from concurrent.futures import ThreadPoolExecutor
from cryptography.fernet import Fernet
import traceback
@log_function_call
def read_image_metadata(image_data: bytes) -> Dict[str, Any]:
    """
    Reads EXIF metadata from an image.

    This function attempts to extract EXIF metadata from the provided image data. It uses the PIL library to open the image
    and then accesses the EXIF information, if available. The extracted metadata is returned as a dictionary.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        Dict[str, Any]: A dictionary containing EXIF metadata, if available.

    Raises:
        ImageOperationError: If reading metadata fails.
    """
    try:
        with Image.open(io.BytesIO(image_data)) as image:
            exif_data = {}
            if 'exif' in image.info:
                exif = image.getexif()
                if exif is not None:
                    for tag, value in exif.items():
                        decoded = ExifTags.TAGS.get(tag, tag)
                        exif_data[decoded] = value
            LoggingManager.debug('Image metadata read successfully.')
            return exif_data
    except Exception as e:
        LoggingManager.error(f'Error reading image metadata: {e}')
        raise ImageOperationError(f'Error reading image metadata due to: {str(e)}') from e
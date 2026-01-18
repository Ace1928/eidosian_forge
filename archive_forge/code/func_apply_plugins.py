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
def apply_plugins(image: Image.Image) -> Image.Image:
    """
    Applies registered image processing plugins to an image.

    This function iterates over the list of registered plugins, applying each plugin's processing method to the image
    in sequence. This allows for the dynamic application of multiple image processing techniques to a single image,
    enhancing its visual appearance or extracting relevant information as required.

    Args:
        image (Image.Image): The input image to be processed by the plugins.

    Returns:
        Image.Image: The image after all registered plugins have been applied.
    """
    for plugin in plugins:
        LoggingManager.debug(f'Applying plugin: {plugin.__class__.__name__}')
        image = plugin.process(image)
    LoggingManager.debug('All plugins applied successfully.')
    return image
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
def test_get_image_hash(self) -> None:
    """
        Tests the image hashing functionality.

        This test verifies that the get_image_hash function correctly generates a hash for the test image data,
        returning a string representation of the hash. The test asserts that the returned hash is a string of length 128,
        indicating a successful hash generation.
        """
    image_hash: str = get_image_hash(self.test_image_data)
    self.assertIsInstance(image_hash, str)
    self.assertEqual(len(image_hash), 128)
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
def test_encryption_key_management(self):
    key1 = EncryptionManager.get_valid_encryption_key()
    key2 = EncryptionManager.get_valid_encryption_key()
    self.assertEqual(key1, key2)
    os.remove(EncryptionManager.KEY_FILE)
    key3 = EncryptionManager.get_valid_encryption_key()
    self.assertNotEqual(key1, key3)
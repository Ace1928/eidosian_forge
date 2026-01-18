import bz2
import gzip
import lzma
import os
import shutil
import struct
import tarfile
import warnings
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union
from .. import config
from ._filelock import FileLock
from .logging import get_logger
@classmethod
def infer_extractor_format(cls, path: Union[Path, str]) -> str:
    magic_number_max_length = cls._get_magic_number_max_length()
    magic_number = cls._read_magic_number(path, magic_number_max_length)
    for extractor_format, extractor in cls.extractors.items():
        if extractor.is_extractable(path, magic_number=magic_number):
            return extractor_format
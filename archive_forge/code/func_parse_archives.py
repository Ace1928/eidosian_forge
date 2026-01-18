import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from .folder import ImageFolder
from .utils import check_integrity, extract_archive, verify_str_arg
def parse_archives(self) -> None:
    if not check_integrity(os.path.join(self.root, META_FILE)):
        parse_devkit_archive(self.root)
    if not os.path.isdir(self.split_folder):
        if self.split == 'train':
            parse_train_archive(self.root)
        elif self.split == 'val':
            parse_val_archive(self.root)
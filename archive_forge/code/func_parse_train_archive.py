import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from .folder import ImageFolder
from .utils import check_integrity, extract_archive, verify_str_arg
def parse_train_archive(root: str, file: Optional[str]=None, folder: str='train') -> None:
    """Parse the train images archive of the ImageNet2012 classification dataset and
    prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the train images archive
        file (str, optional): Name of train images archive. Defaults to
            'ILSVRC2012_img_train.tar'
        folder (str, optional): Optional name for train images folder. Defaults to
            'train'
    """
    archive_meta = ARCHIVE_META['train']
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    _verify_archive(root, file, md5)
    train_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), train_root)
    archives = [os.path.join(train_root, archive) for archive in os.listdir(train_root)]
    for archive in archives:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)
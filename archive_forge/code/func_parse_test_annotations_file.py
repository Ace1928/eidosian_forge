import os
from os.path import abspath, expanduser
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from PIL import Image
from .utils import download_and_extract_archive, download_file_from_google_drive, extract_archive, verify_str_arg
from .vision import VisionDataset
def parse_test_annotations_file(self) -> None:
    filepath = os.path.join(self.root, 'wider_face_split', 'wider_face_test_filelist.txt')
    filepath = abspath(expanduser(filepath))
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            img_path = os.path.join(self.root, 'WIDER_test', 'images', line)
            img_path = abspath(expanduser(img_path))
            self.img_info.append({'img_path': img_path})
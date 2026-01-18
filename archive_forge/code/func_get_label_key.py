import csv
import os
from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Dict, Any
from parlai.core.build_data import download_multiprocess
from parlai.core.params import Opt
from parlai.core.teachers import AbstractImageTeacher
import parlai.utils.typing as PT
def get_label_key(self) -> str:
    return 'question'
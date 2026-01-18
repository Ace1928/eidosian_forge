import glob
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pyarrow as pa
import datasets
import datasets.config
import datasets.data_files
from datasets.naming import filenames_for_dataset_split
def download_and_prepare(self, output_dir: Optional[str]=None, *args, **kwargs):
    if not os.path.exists(self.cache_dir):
        raise ValueError(f"Cache directory for {self.dataset_name} doesn't exist at {self.cache_dir}")
    if output_dir is not None and output_dir != self.cache_dir:
        shutil.copytree(self.cache_dir, output_dir)
import fnmatch
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
from datasets import config
from datasets.commands import BaseDatasetsCLICommand
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadManager
from datasets.download.mock_download_manager import MockDownloadManager
from datasets.load import dataset_module_factory, import_main_class
from datasets.utils.deprecation_utils import deprecated
from datasets.utils.logging import get_logger, set_verbosity_warning
from datasets.utils.py_utils import map_nested
def dummy_data_command_factory(args):
    return DummyDataCommand(args.path_to_dataset, args.auto_generate, args.n_lines, args.json_field, args.xml_tag, args.match_text_files, args.keep_uncompressed, args.cache_dir, args.encoding)
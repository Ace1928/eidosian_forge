import enum
import io
import os
import posixpath
import tarfile
import warnings
import zipfile
from datetime import datetime
from functools import partial
from itertools import chain
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
from .. import config
from ..utils import tqdm as hf_tqdm
from ..utils.deprecation_utils import DeprecatedEnum, deprecated
from ..utils.file_utils import (
from ..utils.info_utils import get_size_checksum_dict
from ..utils.logging import get_logger
from ..utils.py_utils import NestedDataStructure, map_nested, size_str
from ..utils.track import TrackedIterable, tracked_str
from .download_config import DownloadConfig
@staticmethod
def ship_files_with_pipeline(downloaded_path_or_paths, pipeline):
    """Ship the files using Beam FileSystems to the pipeline temp dir.

        Args:
            downloaded_path_or_paths (`str` or `list[str]` or `dict[str, str]`):
                Nested structure containing the
                downloaded path(s).
            pipeline ([`utils.beam_utils.BeamPipeline`]):
                Apache Beam Pipeline.

        Returns:
            `str` or `list[str]` or `dict[str, str]`
        """
    from ..utils.beam_utils import upload_local_to_remote
    remote_dir = pipeline._options.get_all_options().get('temp_location')
    if remote_dir is None:
        raise ValueError("You need to specify 'temp_location' in PipelineOptions to upload files")

    def upload(local_file_path):
        remote_file_path = posixpath.join(remote_dir, config.DOWNLOADED_DATASETS_DIR, os.path.basename(local_file_path))
        logger.info(f'Uploading {local_file_path} ({size_str(os.path.getsize(local_file_path))}) to {remote_file_path}.')
        upload_local_to_remote(local_file_path, remote_file_path)
        return remote_file_path
    uploaded_path_or_paths = map_nested(lambda local_file_path: upload(local_file_path), downloaded_path_or_paths)
    return uploaded_path_or_paths
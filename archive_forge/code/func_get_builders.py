import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile, rmtree
from typing import Generator
import datasets.config
from datasets.builder import DatasetBuilder
from datasets.commands import BaseDatasetsCLICommand
from datasets.download.download_manager import DownloadMode
from datasets.load import dataset_module_factory, import_main_class
from datasets.utils.info_utils import VerificationMode
from datasets.utils.logging import ERROR, get_logger
def get_builders() -> Generator[DatasetBuilder, None, None]:
    if self._all_configs and builder_cls.BUILDER_CONFIGS:
        for i, config in enumerate(builder_cls.BUILDER_CONFIGS):
            if 'config_name' in module.builder_kwargs:
                yield builder_cls(cache_dir=self._cache_dir, data_dir=self._data_dir, **module.builder_kwargs)
            else:
                yield builder_cls(config_name=config.name, cache_dir=self._cache_dir, data_dir=self._data_dir, **module.builder_kwargs)
    elif 'config_name' in module.builder_kwargs:
        yield builder_cls(cache_dir=self._cache_dir, data_dir=self._data_dir, **module.builder_kwargs)
    else:
        yield builder_cls(config_name=config_name, cache_dir=self._cache_dir, data_dir=self._data_dir, **module.builder_kwargs)
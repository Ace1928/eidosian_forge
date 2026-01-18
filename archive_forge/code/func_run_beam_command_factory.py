import os
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile
from typing import List
from datasets import config
from datasets.builder import DatasetBuilder
from datasets.commands import BaseDatasetsCLICommand
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadMode
from datasets.load import dataset_module_factory, import_main_class
from datasets.utils.info_utils import VerificationMode
def run_beam_command_factory(args, **kwargs):
    return RunBeamCommand(args.dataset, args.name, args.cache_dir, args.beam_pipeline_options, args.data_dir, args.all_configs, args.save_info or args.save_infos, args.ignore_verifications, args.force_redownload, **kwargs)
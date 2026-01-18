import logging
import shutil
import sys
from pathlib import Path
from . import EXE_FORMAT
from .config import Config
from .python_helper import PythonExecutable
def create_config_file(dry_run: bool=False, config_file: Path=None, main_file: Path=None):
    """
        Sets up a new pysidedeploy.spec or use an existing config file
    """
    if main_file:
        if main_file.parent != Path.cwd():
            config_file = main_file.parent / 'pysidedeploy.spec'
        else:
            config_file = Path.cwd() / 'pysidedeploy.spec'
    logging.info(f'[DEPLOY] Creating config file {config_file}')
    if not dry_run:
        shutil.copy(Path(__file__).parent / 'default.spec', config_file)
    if dry_run:
        config_file = Path(__file__).parent / 'default.spec'
    return config_file
import logging
import os
import shutil
import subprocess
import hashlib
import json
from typing import Optional, List, Union, Tuple
def delete_conda_env(prefix: str, logger: Optional[logging.Logger]=None) -> bool:
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(f'Deleting conda environment {prefix}')
    conda_path = get_conda_bin_executable('conda')
    delete_cmd = [conda_path, 'remove', '-p', prefix, '--all', '-y']
    exit_code, output = exec_cmd_stream_to_logger(delete_cmd, logger)
    if exit_code != 0:
        logger.debug(f'Failed to delete conda environment {prefix}:\n{output}')
        return False
    return True
import os
import hashlib
import pickle
import time
import shutil
import glob
from ..interfaces.base import BaseInterface
from ..pipeline.engine import Node
from ..pipeline.engine.utils import modify_paths
def rm_all_but(base_dir, dirs_to_keep, warn=False):
    """Remove all the sub-directories of base_dir, but those listed

    Parameters
    ============
    base_dir: string
        The base directory
    dirs_to_keep: set
        The names of the directories to keep
    """
    try:
        all_dirs = os.listdir(base_dir)
    except OSError:
        'Dir has been deleted'
        return
    all_dirs = [d for d in all_dirs if not d.startswith('log.')]
    dirs_to_rm = list(dirs_to_keep.symmetric_difference(all_dirs))
    for dir_name in dirs_to_rm:
        dir_name = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_name):
            if warn:
                print('removing directory: %s' % dir_name)
            shutil.rmtree(dir_name)
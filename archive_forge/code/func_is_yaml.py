import contextlib
import errno
import hashlib
import json
import os
import stat
import tempfile
import time
import yaml
from oslo_utils import excutils
def is_yaml(file_path):
    """Check if file is of yaml type or not.

    This function try to load the input file using yaml.safe_load()
    and return True if loadable. Because every json file can be loadable
    in yaml, so this function return False if file is loadable using
    json.loads() means it is json file.

    :param file_path: The file path to check

    :returns: bool

    """
    with open(file_path, 'r') as fh:
        data = fh.read()
        is_yaml = False
        try:
            json.loads(data)
        except ValueError:
            try:
                yaml.safe_load(data)
                is_yaml = True
            except yaml.scanner.ScannerError:
                pass
        return is_yaml
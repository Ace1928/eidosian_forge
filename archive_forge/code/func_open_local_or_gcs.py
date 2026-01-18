import glob
import logging
import os
import shutil
import subprocess
import time
from apache_beam.io import gcsio
def open_local_or_gcs(path, mode):
    """Opens the given path."""
    if path.startswith('gs://'):
        try:
            return gcsio.GcsIO().open(path, mode)
        except Exception as e:
            logging.error('Retrying after exception reading gcs file: %s', e)
            time.sleep(10)
            return gcsio.GcsIO().open(path, mode)
    else:
        return open(path, mode)
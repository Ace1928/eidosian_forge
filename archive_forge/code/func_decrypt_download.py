from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
from gslib import storage_url
from gslib.utils import execution_util
from gslib.utils import temporary_file_util
from boto import config
def decrypt_download(source_url, destination_url, temporary_file_name, logger):
    """STET-decrypts downloaded file.

  Args:
    source_url (StorageUrl): Copy source.
    destination_url (StorageUrl): Copy destination.
    temporary_file_name (str): Path to temporary file used for download.
    logger (logging.Logger): For logging STET binary output.
  """
    in_file = temporary_file_name
    out_file = temporary_file_util.GetStetTempFileName(destination_url)
    blob_id = source_url.url_string
    _stet_transform(StetSubcommandName.DECRYPT, blob_id, in_file, out_file, logger)
    shutil.move(out_file, in_file)
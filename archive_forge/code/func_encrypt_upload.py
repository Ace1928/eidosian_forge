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
def encrypt_upload(source_url, destination_url, logger):
    """Encrypts a file with STET binary before upload.

  Args:
    source_url (StorageUrl): Copy source.
    destination_url (StorageUrl): Copy destination.
    logger (logging.Logger): For logging STET binary output.

  Returns:
    stet_temporary_file_url (StorageUrl): Path to STET-encrypted file.
  """
    in_file = source_url.object_name
    out_file = temporary_file_util.GetStetTempFileName(source_url)
    blob_id = destination_url.url_string
    _stet_transform(StetSubcommandName.ENCRYPT, blob_id, in_file, out_file, logger)
    return storage_url.StorageUrlFromString(out_file)
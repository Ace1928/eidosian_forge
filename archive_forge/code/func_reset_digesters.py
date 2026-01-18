from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import enum
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
def reset_digesters(digesters):
    """Clears the data from every hash object in a dict of digesters."""
    for hash_algorithm in digesters:
        if hash_algorithm is HashAlgorithm.MD5:
            digesters[hash_algorithm] = hashing.get_md5()
        elif hash_algorithm is HashAlgorithm.CRC32C:
            digesters[hash_algorithm] = fast_crc32c_util.get_crc32c()
        else:
            raise errors.Error('Unknown hash algorithm found in digesters: {}'.format(hash_algorithm))
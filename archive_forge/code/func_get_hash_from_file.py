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
def get_hash_from_file(path, hash_algorithm, start=None, stop=None):
    """Reads file and returns its hash object.

  core.util.files.Checksum does similar things but is different enough to merit
  this function. The primary differences are that this function:
  -Uses a FIPS-safe MD5 object.
  -Accomodates gcloud_crc32c, which uses a Go binary for hashing.
  -Supports start and end index to set byte range for hashing.

  Args:
    path (str): File to read.
    hash_algorithm (HashAlgorithm): Algorithm to hash file with.
    start (int|None): Byte index to start hashing at.
    stop (int|None): Stop hashing at this byte index.

  Returns:
    Hash object for file.
  """
    if hash_algorithm == HashAlgorithm.MD5:
        hash_object = hashing.get_md5()
    elif hash_algorithm == HashAlgorithm.CRC32C:
        hash_object = fast_crc32c_util.get_crc32c()
    else:
        return
    if isinstance(hash_object, fast_crc32c_util.DeferredCrc32c):
        offset = 0 if start is None else start
        length = 0 if stop is None else stop - offset
        hash_object.sum_file(path, offset=offset, length=length)
        return hash_object
    with files.BinaryFileReader(path) as stream:
        if start:
            stream.seek(start)
        while True:
            if stop and stream.tell() >= stop:
                break
            if stop is None or stream.tell() + installers.WRITE_BUFFER_SIZE < stop:
                bytes_to_read = installers.WRITE_BUFFER_SIZE
            else:
                bytes_to_read = stop - stream.tell()
            data = stream.read(bytes_to_read)
            if not data:
                break
            if isinstance(data, str):
                data = data.encode('utf-8')
            hash_object.update(data)
    return hash_object
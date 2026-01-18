import os.path
import struct
import zipfile
import zlib
def countFileChunks(zipinfo, chunksize):
    """
    Count the number of chunks that will result from the given C{ZipInfo}.

    @param zipinfo: a C{zipfile.ZipInfo} instance describing an entry in a zip
    archive to be counted.

    @return: the number of chunks present in the zip file.  (Even an empty file
    counts as one chunk.)
    @rtype: L{int}
    """
    count, extra = divmod(zipinfo.file_size, chunksize)
    if extra > 0:
        count += 1
    return count or 1
import errno
import os
import re
import socket
import time
import six.moves.http_client as httplib
import boto
from boto import config, storage_uri_for_key
from boto.connection import AWSAuthConnection
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.s3.keyfile import KeyFile
from boto.gs.key import Key as GSKey
def get_cur_file_size(fp, position_to_eof=False):
    """
    Returns size of file, optionally leaving fp positioned at EOF.
    """
    if isinstance(fp, KeyFile) and (not position_to_eof):
        return fp.getkey().size
    if not position_to_eof:
        cur_pos = fp.tell()
    fp.seek(0, os.SEEK_END)
    cur_file_size = fp.tell()
    if not position_to_eof:
        fp.seek(cur_pos, os.SEEK_SET)
    return cur_file_size
import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
def get_pack_file(req, backend, mat):
    req.cache_forever()
    path = _url_to_path(mat.group())
    logger.info('Sending pack file %s', path)
    return send_file(req, get_repo(backend, mat).get_named_file(path), 'application/x-git-packed-objects')
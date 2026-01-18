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
def get_info_packs(req, backend, mat):
    req.nocache()
    req.respond(HTTP_OK, 'text/plain')
    logger.info('Emulating dumb info/packs')
    return generate_objects_info_packs(get_repo(backend, mat))
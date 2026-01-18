import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
def compute_send(self, src):
    sendrefs = dict(src.get_refs())
    del sendrefs[b'HEAD']
    return (sendrefs, src.generate_pack_data)
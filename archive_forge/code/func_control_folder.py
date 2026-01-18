from __future__ import annotations
import os
import shutil
import tempfile
import uuid
from collections import namedtuple
from pathlib import Path
from queue import Empty
from time import monotonic
from kombu.exceptions import ChannelError
from kombu.transport import virtual
from kombu.utils.encoding import bytes_to_str, str_to_bytes
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
@property
def control_folder(self):
    return Path(self.transport_options.get('control_folder', 'control'))
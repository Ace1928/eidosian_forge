import os
import stat
import textwrap
from email import message_from_file
from email.message import Message
from tempfile import NamedTemporaryFile
from typing import Optional, List
from distutils.util import rfc822_escape
from . import _normalization, _reqs
from .extern.packaging.markers import Marker
from .extern.packaging.requirements import Requirement
from .extern.packaging.version import Version
from .warnings import SetuptoolsDeprecationWarning
def get_metadata_version(self):
    mv = getattr(self, 'metadata_version', None)
    if mv is None:
        mv = Version('2.1')
        self.metadata_version = mv
    return mv
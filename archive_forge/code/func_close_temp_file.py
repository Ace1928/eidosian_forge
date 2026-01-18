from __future__ import absolute_import, division, print_function
import os
from tempfile import TemporaryFile, NamedTemporaryFile
from select import select
from abc import ABCMeta, abstractmethod
import tarfile
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
def close_temp_file(self):
    if self.named_temp_file:
        self.named_temp_file.close()
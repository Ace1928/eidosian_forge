import errno
import glob
import hashlib
import importlib.metadata as importlib_metadata
import itertools
import json
import logging
import os
import os.path
import struct
import sys
def get_group_all(self, group, path=None):
    result = []
    data = self._get_data_for_path(path)
    group_data = data.get('groups', {}).get(group, [])
    for vals in group_data:
        result.append(importlib_metadata.EntryPoint(*vals))
    return result
import argparse
import asyncio
from datetime import datetime
import importlib
import inspect  # pylint: disable=syntax-error
import io
import json
import collections  # pylint: disable=syntax-error
import os
import signal
import sys
import traceback
import zipfile
from zipimport import zipimporter
import pickle
import uuid
import ansible.module_utils.basic
def find_module_name(self):
    with zipfile.ZipFile(self.ansiblez_path) as zip:
        for path in zip.namelist():
            if not path.startswith('ansible_collections'):
                continue
            if not path.endswith('.py'):
                continue
            if path.endswith('__init__.py'):
                continue
            splitted = path.split('/')
            if len(splitted) != 6:
                continue
            if splitted[-3:-1] != ['plugins', 'modules']:
                continue
            collection = '.'.join(splitted[1:3])
            name = splitted[-1][:-3]
            return (collection, name)
    raise Exception('Cannot find module name')
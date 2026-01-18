import doctest
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import ClassVar, List
from unittest import SkipTest, expectedFailure, skipIf
from unittest import TestCase as _TestCase
def self_test_suite():
    names = ['archive', 'blackbox', 'bundle', 'client', 'config', 'credentials', 'diff_tree', 'fastexport', 'file', 'grafts', 'graph', 'greenthreads', 'hooks', 'ignore', 'index', 'lfs', 'line_ending', 'lru_cache', 'mailmap', 'objects', 'objectspec', 'object_store', 'missing_obj_finder', 'pack', 'patch', 'porcelain', 'protocol', 'reflog', 'refs', 'repository', 'server', 'stash', 'utils', 'walk', 'web']
    module_names = ['dulwich.tests.test_' + name for name in names]
    loader = unittest.TestLoader()
    return loader.loadTestsFromNames(module_names)
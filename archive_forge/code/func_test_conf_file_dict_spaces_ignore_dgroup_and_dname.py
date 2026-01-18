import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
def test_conf_file_dict_spaces_ignore_dgroup_and_dname(self):
    self._do_dgroup_and_dname_test_ignore(cfg.DictOpt, 'k1:d,k2:e,k3:f', {'k1': 'd', 'k2': 'e', 'k3': 'f'})
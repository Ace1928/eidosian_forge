import argparse as argparse_mod
import collections
import copy
import errno
import json
import os
import re
import sys
import typing as ty
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import loading
import platformdirs
import yaml
from openstack import _log
from openstack.config import _util
from openstack.config import cloud_region
from openstack.config import defaults
from openstack.config import vendors
from openstack import exceptions
from openstack import warnings as os_warnings
def option_prompt(self, config, p_opt):
    """Prompt user for option that requires a value"""
    if getattr(p_opt, 'prompt', None) is not None and p_opt.dest not in config['auth'] and (self._pw_callback is not None):
        config['auth'][p_opt.dest] = self._pw_callback(p_opt.prompt)
    return config
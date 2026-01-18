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
@staticmethod
def set_one_cloud(config_file, cloud, set_config=None):
    """Set a single cloud configuration.

        :param string config_file:
            The path to the config file to edit. If this file does not exist
            it will be created.
        :param string cloud:
            The name of the configuration to save to clouds.yaml
        :param dict set_config: Configuration options to be set
        """
    set_config = set_config or {}
    cur_config = {}
    try:
        with open(config_file) as fh:
            cur_config = yaml.safe_load(fh)
    except IOError as e:
        if e.errno != 2:
            raise
        pass
    clouds_config = cur_config.get('clouds', {})
    cloud_config = _auth_update(clouds_config.get(cloud, {}), set_config)
    clouds_config[cloud] = cloud_config
    cur_config['clouds'] = clouds_config
    with open(config_file, 'w') as fh:
        yaml.safe_dump(cur_config, fh, default_flow_style=False)
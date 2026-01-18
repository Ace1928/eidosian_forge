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
def get_one_cloud_osc(self, cloud=None, validate=True, argparse=None, **kwargs):
    """Retrieve a single CloudRegion and merge additional options

        :param string cloud:
            The name of the configuration to load from clouds.yaml
        :param boolean validate:
            Validate the config. Setting this to False causes no auth plugin
            to be created. It's really only useful for testing.
        :param Namespace argparse:
            An argparse Namespace object; allows direct passing in of
            argparse options to be added to the cloud config.  Values
            of None and '' will be removed.
        :param region_name: Name of the region of the cloud.
        :param kwargs: Additional configuration options

        :raises: keystoneauth1.exceptions.MissingRequiredOptions
            on missing required auth parameters
        """
    args = self._fix_args(kwargs, argparse=argparse)
    if cloud is None:
        if 'cloud' in args:
            cloud = args['cloud']
        else:
            cloud = self.default_cloud
    config = self._get_base_cloud_config(cloud)
    if 'region_name' not in args:
        args['region_name'] = ''
    region = self._get_region(cloud=cloud, region_name=args['region_name'])
    args['region_name'] = region['name']
    region_args = copy.deepcopy(region['values'])
    config.pop('regions', None)
    for arg_list in (region_args, args):
        for key, val in iter(arg_list.items()):
            if val is not None:
                if key == 'auth' and config[key] is not None:
                    config[key] = _auth_update(config[key], val)
                else:
                    config[key] = val
    config = self.magic_fixes(config)
    config = self.auth_config_hook(config)
    if validate:
        loader = self._get_auth_loader(config)
        config = self._validate_auth_correctly(config, loader)
        auth_plugin = loader.load_from_options(**config['auth'])
    else:
        auth_plugin = None
    for key, value in config.items():
        if hasattr(value, 'format') and key not in FORMAT_EXCLUSIONS:
            config[key] = value.format(**config)
    force_ipv4 = config.pop('force_ipv4', self.force_ipv4)
    prefer_ipv6 = config.pop('prefer_ipv6', True)
    if not prefer_ipv6:
        force_ipv4 = True
    if cloud is None:
        cloud_name = ''
    else:
        cloud_name = str(cloud)
    return self._cloud_region_class(name=cloud_name, region_name=config['region_name'], config=config, extra_config=self.extra_config, force_ipv4=force_ipv4, auth_plugin=auth_plugin, openstack_config=self, cache_auth=self._cache_auth, cache_expiration_time=self._cache_expiration_time, cache_expirations=self._cache_expirations, cache_path=self._cache_path, cache_class=self._cache_class, cache_arguments=self._cache_arguments, password_callback=self._pw_callback)
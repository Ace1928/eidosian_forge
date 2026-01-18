from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import base64
import errno
import io
import json
import logging
import os
import subprocess
from containerregistry.client import docker_name
import httplib2
from oauth2client import client as oauth2client
import six
def setCustomConfigDir(self, config_dir):
    if not os.path.isdir(config_dir):
        raise Exception('Attempting to override docker configuration directory to invalid directory: {}'.format(config_dir))
    self._config_dir = config_dir
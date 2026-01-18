from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.urls import Request, SSLValidationError, ConnectionError
from ansible.module_utils.parsing.convert_bool import boolean as strtobool
from ansible.module_utils.six import PY2
from ansible.module_utils.six import raise_from, string_types
from ansible.module_utils.six.moves import StringIO
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.http_cookiejar import CookieJar
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode, quote
from ansible.module_utils.six.moves.configparser import ConfigParser, NoOptionError
from socket import getaddrinfo, IPPROTO_TCP
import time
import re
from json import loads, dumps
from os.path import isfile, expanduser, split, join, exists, isdir
from os import access, R_OK, getcwd, environ
def load_config_files(self):
    config_files = ['/etc/tower/tower_cli.cfg', join(expanduser('~'), '.{0}'.format(self.config_name))]
    local_dir = getcwd()
    config_files.append(join(local_dir, self.config_name))
    while split(local_dir)[1]:
        local_dir = split(local_dir)[0]
        config_files.insert(2, join(local_dir, '.{0}'.format(self.config_name)))
    if self.params.get('controller_config_file'):
        duplicated_params = [fn for fn in self.AUTH_ARGSPEC if fn != 'controller_config_file' and self.params.get(fn) is not None]
        if duplicated_params:
            self.warn('The parameter(s) {0} were provided at the same time as controller_config_file. Precedence may be unstable, we suggest either using config file or params.'.format(', '.join(duplicated_params)))
        try:
            self.load_config(self.params.get('controller_config_file'))
        except ConfigFileException as cfe:
            self.fail_json(msg=cfe)
    else:
        for config_file in config_files:
            if exists(config_file) and (not isdir(config_file)):
                try:
                    self.load_config(config_file)
                except ConfigFileException:
                    self.fail_json(msg='The config file {0} is not properly formatted'.format(config_file))
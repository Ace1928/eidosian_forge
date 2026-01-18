from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.basic import missing_required_lib
def parse_from_mysql_config_file(cnf):
    cp = configparser.ConfigParser()
    cp.read(cnf)
    return cp
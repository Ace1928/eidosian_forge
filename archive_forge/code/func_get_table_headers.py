import configparser as config_parser
import os
from tempest.lib.cli import base
def get_table_headers(self, action, flags='', params=''):
    output = self._zun(action=action, flags=flags, params=params)
    table = self.parser.table(output)
    return table['headers']
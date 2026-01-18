import os
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions as tempest_exc
def get_template_path(self, templ_name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../templates/%s' % templ_name)
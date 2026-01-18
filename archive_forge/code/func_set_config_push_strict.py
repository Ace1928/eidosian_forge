import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def set_config_push_strict(self, value):
    br = branch.Branch.open('local')
    br.get_config_stack().set('push_strict', value)
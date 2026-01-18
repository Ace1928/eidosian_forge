import os, sys
import re
import logging; log = logging.getLogger(__name__)
from passlib.utils.compat import print_
def do_setup_gae(path, runtime):
    """write fake GAE ``app.yaml`` to current directory so nosegae will work"""
    from passlib.tests.utils import set_file
    set_file(os.path.join(path, 'app.yaml'), 'application: fake-app\nversion: 2\nruntime: %s\napi_version: 1\nthreadsafe: no\n\nhandlers:\n- url: /.*\n  script: dummy.py\n\nlibraries:\n- name: django\n  version: "latest"\n' % runtime)
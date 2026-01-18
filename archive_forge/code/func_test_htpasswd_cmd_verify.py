from __future__ import with_statement
from logging import getLogger
import os
import subprocess
from passlib import apache, registry
from passlib.exc import MissingBackendError
from passlib.utils.compat import irange
from passlib.tests.backports import unittest
from passlib.tests.utils import TestCase, get_file, set_file, ensure_mtime_changed
from passlib.utils.compat import u
from passlib.utils import to_bytes
from passlib.utils.handlers import to_unicode_for_identify
@requires_htpasswd_cmd
def test_htpasswd_cmd_verify(self):
    """
        verify "htpasswd" command can read output
        """
    path = self.mktemp()
    ht = apache.HtpasswdFile(path=path, new=True)

    def hash_scheme(pwd, scheme):
        return ht.context.handler(scheme).hash(pwd)
    ht.set_hash('user1', hash_scheme('password', 'apr_md5_crypt'))
    host_no_bcrypt = apache.htpasswd_defaults['host_apache_22']
    ht.set_hash('user2', hash_scheme('password', host_no_bcrypt))
    host_best = apache.htpasswd_defaults['host']
    ht.set_hash('user3', hash_scheme('password', host_best))
    ht.set_hash('user4', '$xxx$foo$bar$baz')
    ht.save()
    self.assertFalse(_call_htpasswd_verify(path, 'user1', 'wrong'))
    self.assertFalse(_call_htpasswd_verify(path, 'user2', 'wrong'))
    self.assertFalse(_call_htpasswd_verify(path, 'user3', 'wrong'))
    self.assertFalse(_call_htpasswd_verify(path, 'user4', 'wrong'))
    self.assertTrue(_call_htpasswd_verify(path, 'user1', 'password'))
    self.assertTrue(_call_htpasswd_verify(path, 'user2', 'password'))
    self.assertTrue(_call_htpasswd_verify(path, 'user3', 'password'))
import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_cyclical_protection(self):
    self.secret_cls.authorized = True
    self.deepsecret_cls.authorized = True
    response = self.app.get('/secret/1/deepsecret/2/deepsecret/')
    assert response.status_int == 200
    assert response.body == b'Deep Secret'
    assert 'secretcontroller' in self.permissions_checked
    assert 'deepsecret' in self.permissions_checked
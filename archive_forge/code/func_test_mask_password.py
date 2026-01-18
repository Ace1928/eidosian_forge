import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_mask_password(self):
    payload = "test = 'password'  :   'aaaaaa'"
    expected = "test = 'password'  :   '111'"
    self.assertEqual(expected, strutils.mask_password(payload, secret='111'))
    payload = 'mysqld --password "aaaaaa"'
    expected = 'mysqld --password "****"'
    self.assertEqual(expected, strutils.mask_password(payload, secret='****'))
    payload = 'mysqld --password aaaaaa'
    expected = 'mysqld --password ???'
    self.assertEqual(expected, strutils.mask_password(payload, secret='???'))
    payload = 'mysqld --password = "aaaaaa"'
    expected = 'mysqld --password = "****"'
    self.assertEqual(expected, strutils.mask_password(payload, secret='****'))
    payload = "mysqld --password = 'aaaaaa'"
    expected = "mysqld --password = '****'"
    self.assertEqual(expected, strutils.mask_password(payload, secret='****'))
    payload = 'mysqld --password = aaaaaa'
    expected = 'mysqld --password = ****'
    self.assertEqual(expected, strutils.mask_password(payload, secret='****'))
    payload = 'test = password =   aaaaaa'
    expected = 'test = password =   111'
    self.assertEqual(expected, strutils.mask_password(payload, secret='111'))
    payload = 'test = password=   aaaaaa'
    expected = 'test = password=   111'
    self.assertEqual(expected, strutils.mask_password(payload, secret='111'))
    payload = 'test = password =aaaaaa'
    expected = 'test = password =111'
    self.assertEqual(expected, strutils.mask_password(payload, secret='111'))
    payload = 'test = password=aaaaaa'
    expected = 'test = password=111'
    self.assertEqual(expected, strutils.mask_password(payload, secret='111'))
    payload = 'test = "original_password" : "aaaaaaaaa"'
    expected = 'test = "original_password" : "***"'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'test = "param1" : "value"'
    expected = 'test = "param1" : "value"'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'test = "original_password" : "aaaaa"aaaa"'
    expected = 'test = "original_password" : "***"'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "{'adminPass':'TL0EfN33'}"
    payload = str(payload)
    expected = "{'adminPass':'***'}"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "{'adminPass':'TL0E'fN33'}"
    payload = str(payload)
    expected = "{'adminPass':'***'}"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "{'token':'mytoken'}"
    payload = str(payload)
    expected = "{'token':'***'}"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "test = 'node.session.auth.password','-v','TL0EfN33','nomask'"
    expected = "test = 'node.session.auth.password','-v','***','nomask'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "test = 'node.session.auth.password', '--password', 'TL0EfN33', 'nomask'"
    expected = "test = 'node.session.auth.password', '--password', '***', 'nomask'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "test = 'node.session.auth.password', '--password', 'TL0EfN33'"
    expected = "test = 'node.session.auth.password', '--password', '***'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'test = node.session.auth.password -v TL0EfN33 nomask'
    expected = 'test = node.session.auth.password -v *** nomask'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'test = node.session.auth.password --password TL0EfN33 nomask'
    expected = 'test = node.session.auth.password --password *** nomask'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'test = node.session.auth.password --password TL0EfN33'
    expected = 'test = node.session.auth.password --password ***'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'test = cmd --password myé\x80\x80pass'
    expected = 'test = cmd --password ***'
    self.assertEqual(expected, strutils.mask_password(payload))
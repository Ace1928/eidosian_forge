import copy
import gc
import os
import sys
import time
from queue import Queue
from threading import Event, Thread
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest
@mark.skipif(mock is None, reason='requires unittest.mock')
def test_mockable(self):
    m = mock.Mock(spec=self.context)
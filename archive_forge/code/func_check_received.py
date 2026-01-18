import os
import requests
import subprocess
import time
import uuid
import concurrent.futures
from oslo_config import cfg
from testtools import matchers
import oslo_messaging
from oslo_messaging.tests.functional import utils
def check_received(listener, publisher, messages):
    actuals = sorted([listener.events.get(timeout=get_timeout) for __ in range(len(a_out))])
    expected = sorted([['info', m[0], m[1], publisher] for m in messages])
    self.assertEqual(expected, actuals)
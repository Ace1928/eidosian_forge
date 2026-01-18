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
def test_all_categories(self):
    get_timeout = 1
    if self.notify_url.startswith('kafka://'):
        get_timeout = 5
        self.conf.set_override('consumer_group', 'test_all_categories', group='oslo_messaging_kafka')
    listener = self.useFixture(utils.NotificationFixture(self.conf, self.notify_url, ['test_all_categories']))
    n = listener.notifier('abc')
    cats = ['debug', 'audit', 'info', 'warn', 'error', 'critical']
    events = [(getattr(n, c), c, 'type-' + c, c + '-data') for c in cats]
    for e in events:
        e[0]({}, e[2], e[3])
    received = {}
    for expected in events:
        e = listener.events.get(timeout=get_timeout)
        received[e[0]] = e
    for expected in events:
        actual = received[expected[1]]
        self.assertEqual(expected[1], actual[0])
        self.assertEqual(expected[2], actual[1])
        self.assertEqual(expected[3], actual[2])
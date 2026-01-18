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
def test_multiple_topics(self):
    get_timeout = 1
    if self.notify_url.startswith('kafka://'):
        get_timeout = 5
        self.conf.set_override('consumer_group', 'test_multiple_topics', group='oslo_messaging_kafka')
    listener = self.useFixture(utils.NotificationFixture(self.conf, self.notify_url, ['a', 'b']))
    a = listener.notifier('pub-a', topics=['a'])
    b = listener.notifier('pub-b', topics=['b'])
    sent = {'pub-a': [a, 'test-a', 'payload-a'], 'pub-b': [b, 'test-b', 'payload-b']}
    for e in sent.values():
        e[0].info({}, e[1], e[2])
    received = {}
    while len(received) < len(sent):
        e = listener.events.get(timeout=get_timeout)
        received[e[3]] = e
    for key in received:
        actual = received[key]
        expected = sent[key]
        self.assertEqual('info', actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertEqual(expected[2], actual[2])
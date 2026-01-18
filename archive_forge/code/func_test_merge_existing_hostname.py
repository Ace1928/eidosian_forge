import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_merge_existing_hostname(self):
    url = 'http://www.yahoo.com/'
    parsed = misc.parse_uri(url)
    joined = misc.merge_uri(parsed, {'hostname': 'b.com'})
    self.assertEqual('b.com', joined.get('hostname'))
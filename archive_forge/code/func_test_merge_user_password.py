import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_merge_user_password(self):
    url = 'http://josh:harlow@www.yahoo.com/'
    parsed = misc.parse_uri(url)
    joined = misc.merge_uri(parsed, {})
    self.assertEqual('www.yahoo.com', joined.get('hostname'))
    self.assertEqual('josh', joined.get('username'))
    self.assertEqual('harlow', joined.get('password'))
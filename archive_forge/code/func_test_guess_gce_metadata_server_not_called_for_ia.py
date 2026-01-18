import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
def test_guess_gce_metadata_server_not_called_for_ia(self):
    with mock.patch.object(GoogleAuthType, '_is_gce', return_value=False):
        self.assertEqual(GoogleAuthType._is_gce.call_count, 0)
        self.assertEqual(GoogleAuthType.guess_type(GCE_PARAMS_IA_2[0]), GoogleAuthType.IA)
        self.assertEqual(GoogleAuthType._is_gce.call_count, 0)
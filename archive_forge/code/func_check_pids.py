import http.client as http
import os
import re
import time
import psutil
import requests
from glance.tests import functional
from glance.tests.utils import execute
def check_pids(pre, post=None, workers=2):
    if post is None:
        if len(pre) == workers:
            return True
        else:
            return False
    if len(post) == workers:
        if post.intersection(pre) == set():
            return True
    return False
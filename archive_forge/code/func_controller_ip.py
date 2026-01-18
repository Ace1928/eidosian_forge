from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
@controller_ip.setter
def controller_ip(self, controller_ip):
    self.avi_credentials.controller = controller_ip
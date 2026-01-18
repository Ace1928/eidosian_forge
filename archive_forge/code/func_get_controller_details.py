from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def get_controller_details(self):
    result = {'controller_ip': self.controller_ip, 'controller_api_version': self.remote_api_version}
    return result
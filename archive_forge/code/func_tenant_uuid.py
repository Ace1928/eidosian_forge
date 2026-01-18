from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
@tenant_uuid.setter
def tenant_uuid(self, tenant_uuid):
    self.avi_credentials.tenant_uuid = tenant_uuid
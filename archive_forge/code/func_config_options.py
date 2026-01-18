import json
import os
import sys
from troveclient.compat import common
def config_options(oparser):
    oparser.add_option('-u', '--url', default='http://localhost:5000/v1.1', help='Auth API endpoint URL with port and version.                             Default: http://localhost:5000/v1.1')
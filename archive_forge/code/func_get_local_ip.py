from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def get_local_ip(self):
    """Return the local ip address of the Google Cloud VM the workload is running on."""
    return _request_compute_metadata('instance/network-interfaces/0/ip')
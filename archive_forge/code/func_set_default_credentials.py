import time
from functools import wraps
from boto.swf.layer1 import Layer1
from boto.swf.layer1_decisions import Layer1Decisions
def set_default_credentials(aws_access_key_id, aws_secret_access_key):
    """Set default credentials."""
    DEFAULT_CREDENTIALS.update({'aws_access_key_id': aws_access_key_id, 'aws_secret_access_key': aws_secret_access_key})
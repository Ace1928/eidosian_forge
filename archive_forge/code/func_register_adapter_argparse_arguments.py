import os
import warnings
import requests
from keystoneauth1 import _fair_semaphore
from keystoneauth1 import session
def register_adapter_argparse_arguments(*args, **kwargs):
    return Adapter.register_argparse_arguments(*args, **kwargs)
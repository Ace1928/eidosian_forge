import base64
import binascii
import calendar
import concurrent.futures
import datetime
import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import urllib3
from blobfile import _common as common
from blobfile import _xml as xml
from blobfile._common import (
def load_subscription_ids() -> List[str]:
    """
    Return a list of subscription ids from the local azure profile
    the default subscription will appear first in the list
    """
    default_profile_path = os.path.expanduser('~/.azure/azureProfile.json')
    if not os.path.exists(default_profile_path):
        return []
    with open(default_profile_path, 'rb') as f:
        profile = json.loads(f.read().decode('utf-8-sig'))
    subscriptions = profile.get('subscriptions', [])

    def key_fn(x: Mapping[str, Any]) -> bool:
        return x['isDefault']
    subscriptions.sort(key=key_fn, reverse=True)
    return [sub['id'] for sub in subscriptions]
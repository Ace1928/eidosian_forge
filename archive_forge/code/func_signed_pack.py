import base64
import hashlib
import hmac
import json
import os
import uuid
from oslo_utils import secretutils
from oslo_utils import uuidutils
def signed_pack(data, hmac_key):
    """Pack and sign data with hmac_key."""
    raw_data = base64.urlsafe_b64encode(binary_encode(json.dumps(data)))
    return (raw_data, generate_hmac(raw_data, hmac_key) if hmac_key else None)
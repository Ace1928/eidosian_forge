from __future__ import absolute_import, division, print_function
import base64
import hashlib
import json
import re
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
def get_challenge_data(self, client):
    """
        Returns a dict with the data for all proposed (and supported) challenges
        of the given authorization.
        """
    data = {}
    for challenge in self.challenges:
        validation_data = challenge.get_validation_data(client, self.identifier_type, self.identifier)
        if validation_data is not None:
            data[challenge.type] = validation_data
    return data
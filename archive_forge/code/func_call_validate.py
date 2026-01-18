from __future__ import absolute_import, division, print_function
import base64
import hashlib
import json
import re
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
def call_validate(self, client, challenge_type, wait=True):
    """
        Validate the authorization provided in the auth dict. Returns True
        when the validation was successful and False when it was not.
        """
    challenge = self.find_challenge(challenge_type)
    if challenge is None:
        raise ModuleFailException('Found no challenge of type "{challenge}" for identifier {identifier}!'.format(challenge=challenge_type, identifier=self.combined_identifier))
    challenge.call_validate(client)
    if not wait:
        return self.status == 'valid'
    return self.wait_for_validation(client, challenge_type)
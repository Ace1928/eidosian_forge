import base64
import json
import os
import unittest
import mock
from google_reauth import challenges, errors
import pyu2f
def testSaml(self):
    metadata = {'status': 'READY', 'challengeId': 1, 'challengeType': 'SAML', 'securityKey': {}}
    challenge = challenges.SamlChallenge()
    self.assertEqual(True, challenge.is_locally_eligible)
    with self.assertRaises(errors.ReauthSamlLoginRequiredError):
        challenge.obtain_challenge_input(metadata)
import base64
import json
import os
import unittest
import mock
from google_reauth import challenges, errors
import pyu2f
@mock.patch('getpass.getpass', return_value=None)
def testNoPassword(self, getpass_mock):
    self.assertEqual(challenges.PasswordChallenge().obtain_challenge_input({}), {'credential': ' '})
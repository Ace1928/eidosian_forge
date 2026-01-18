from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_or_create_account(self, rest_api):
    """
        Get Account
        :return: Account ID
        """
    accounts, error = self.get_account_info(rest_api)
    if error is not None:
        return (None, error)
    if len(accounts) == 0:
        return (None, 'Error: account cannot be located - check credentials or provide account_id.')
    return (accounts[0]['accountPublicId'], None)
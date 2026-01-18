import logging
import os
import shutil
import sys
import urllib.parse
from typing import (
from pip._internal.cli.spinners import SpinnerInterface
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import (
from pip._internal.utils.urls import get_url_scheme
def get_url_rev_options(self, url: HiddenText) -> Tuple[HiddenText, RevOptions]:
    """
        Return the URL and RevOptions object to use in obtain(),
        as a tuple (url, rev_options).
        """
    secret_url, rev, user_pass = self.get_url_rev_and_auth(url.secret)
    username, secret_password = user_pass
    password: Optional[HiddenText] = None
    if secret_password is not None:
        password = hide_value(secret_password)
    extra_args = self.make_rev_args(username, password)
    rev_options = self.make_rev_options(rev, extra_args=extra_args)
    return (hide_url(secret_url), rev_options)
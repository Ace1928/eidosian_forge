import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def set_date(self, date):
    """ set the date of the top changelog entry

        :param date: str
            a properly formatted date string (`date -R` format; see Policy)
        """
    self._blocks[0].date = date
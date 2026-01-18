import subprocess
import tempfile
from ... import errors
from ... import revision as _mod_revision
from ...config import ListOption, Option, bool_from_store, int_from_store
from ...email_message import EmailMessage
from ...smtp_connection import SMTPConnection
def extra_headers(self):
    """Additional headers to include when sending."""
    result = {}
    headers = self.config.get('revision_mail_headers')
    if not headers:
        return
    for line in headers:
        key, value = line.split(': ', 1)
        result[key] = value
    return result
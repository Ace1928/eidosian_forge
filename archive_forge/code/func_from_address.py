import subprocess
import tempfile
from ... import errors
from ... import revision as _mod_revision
from ...config import ListOption, Option, bool_from_store, int_from_store
from ...email_message import EmailMessage
from ...smtp_connection import SMTPConnection
def from_address(self):
    """What address should I send from."""
    result = self.config.get('post_commit_sender')
    if result is None:
        result = self.config.get('email')
    return result
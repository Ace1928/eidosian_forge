import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
@classmethod
def cleanup_tsigkey(cls, client, tsigkey_id):
    try:
        client.tsigkey_delete(tsigkey_id)
    except CommandFailed:
        pass
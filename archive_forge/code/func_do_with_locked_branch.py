import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
def do_with_locked_branch(self, branch, location):
    branch._set_parent_location(location.decode('utf-8'))
    return SuccessfulSmartServerResponse(())
import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
def do_bzrdir_request(self):
    """Open a branch at path and return the reference or format.

        This version introduced in 2.1.

        Differences to SmartServerRequestOpenBranchV2:
          * can return 2-element ('nobranch', extra), where 'extra' is a string
            with an explanation like 'location is a repository'.  Previously
            a 'nobranch' response would never have more than one element.
        """
    try:
        reference_url = self._bzrdir.get_branch_reference()
        if reference_url is None:
            br = self._bzrdir.open_branch(ignore_fallbacks=True)
            format = br._format.network_name()
            return SuccessfulSmartServerResponse((b'branch', format))
        else:
            return SuccessfulSmartServerResponse((b'ref', reference_url.encode('utf-8')))
    except errors.NotBranchError as e:
        str(e)
        resp = (b'nobranch',)
        detail = e.detail
        if detail:
            if detail.startswith(': '):
                detail = detail[2:]
            resp += (detail.encode('utf-8'),)
        return FailedSmartServerResponse(resp)
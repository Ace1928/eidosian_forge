import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
def check_and_follow_branch_reference(self, url):
    """Check URL (and possibly the referenced URL).

        This method checks that `url` passes the policy's `check_one_url`
        method, and if `url` refers to a branch reference, it checks whether
        references are allowed and whether the reference's URL passes muster
        also -- recursively, until a real branch is found.

        :param url: URL to check
        :raise BranchLoopError: If the branch references form a loop.
        :raise BranchReferenceForbidden: If this opener forbids branch
            references.
        """
    while True:
        if url in self._seen_urls:
            raise BranchLoopError()
        self._seen_urls.add(url)
        self.policy.check_one_url(url)
        next_url = self.follow_reference(url)
        if next_url is None:
            return url
        url = next_url
        if not self.policy.should_follow_references():
            raise BranchReferenceForbidden(url)
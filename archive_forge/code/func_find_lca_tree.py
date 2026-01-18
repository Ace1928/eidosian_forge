import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
def find_lca_tree(self, other):
    """Find the revision tree for the LCA of this branch and other.

        :param other: Another LaunchpadBranch
        :return: The RevisionTree of the LCA of this branch and other.
        """
    graph = self.bzr.repository.get_graph(other.bzr.repository)
    lca = graph.find_unique_lca(self.bzr.last_revision(), other.bzr.last_revision())
    return self.bzr.repository.revision_tree(lca)
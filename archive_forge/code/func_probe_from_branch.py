import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
@classmethod
def probe_from_branch(cls, branch):
    """Create a Forge object if this forge knows about a branch."""
    url = urlutils.strip_segment_parameters(branch.user_url)
    return cls.probe_from_url(url, possible_transports=[branch.control_transport])
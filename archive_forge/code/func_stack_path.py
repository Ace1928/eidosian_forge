import collections
import re
from oslo_utils import encodeutils
from urllib import parse as urlparse
from heat.common.i18n import _
def stack_path(self):
    """Return a URL-encoded path segment of a URL without a tenant.

        Returned in the form:
            <stack_name>/<stack_id>
        """
    return '%s/%s' % (urlparse.quote(self.stack_name, ''), urlparse.quote(self.stack_id, ''))
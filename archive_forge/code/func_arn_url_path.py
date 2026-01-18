import collections
import re
from oslo_utils import encodeutils
from urllib import parse as urlparse
from heat.common.i18n import _
def arn_url_path(self):
    """Return an ARN quoted correctly for use in a URL."""
    return '/' + urlparse.quote(self.arn())
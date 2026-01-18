import collections
import re
from oslo_utils import encodeutils
from urllib import parse as urlparse
from heat.common.i18n import _
@classmethod
def from_arn(cls, arn):
    """Generate a new HeatIdentifier by parsing the supplied ARN."""
    fields = arn.split(':')
    if len(fields) < 6 or fields[0].lower() != 'arn':
        raise ValueError(_('"%s" is not a valid ARN') % arn)
    id_fragment = ':'.join(fields[5:])
    path = cls.path_re.match(id_fragment)
    if fields[1] != 'openstack' or fields[2] != 'heat' or (not path):
        raise ValueError(_('"%s" is not a valid Heat ARN') % arn)
    return cls(urlparse.unquote(fields[4]), urlparse.unquote(path.group(1)), urlparse.unquote(path.group(2)), urlparse.unquote(path.group(3)))
from urllib import parse
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
def tags_to_headers(self):
    if self.properties[self.TAGS] is None:
        return {}
    return dict((('X-Container-Meta-S3-Tag-' + tm[self.TAG_KEY], tm[self.TAG_VALUE]) for tm in self.properties[self.TAGS]))
import glance_store
from glance.api import policy
from glance.api import property_protections
from glance.common import property_utils
from glance.common import store_utils
import glance.db
import glance.domain
import glance.location
import glance.notifier
import glance.quota
def get_metadef_tag_factory(self, context):
    factory = glance.domain.MetadefTagFactory()
    factory = glance.notifier.MetadefTagFactoryProxy(factory, context, self.notifier)
    return factory
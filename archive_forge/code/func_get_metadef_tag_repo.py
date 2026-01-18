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
def get_metadef_tag_repo(self, context):
    """Get the layered MetadefTagRepo model.

        This is where we construct the "the onion" by layering
        MetadefTagRepo models on top of each other, starting with
        the DB at the bottom.

        :param context: The RequestContext
        :returns: An MetadefTagRepo-like object
        """
    repo = glance.db.MetadefTagRepo(context, self.db_api)
    repo = glance.notifier.MetadefTagRepoProxy(repo, context, self.notifier)
    return repo
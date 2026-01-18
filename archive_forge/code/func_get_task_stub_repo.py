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
def get_task_stub_repo(self, context):
    repo = glance.db.TaskRepo(context, self.db_api)
    repo = glance.notifier.TaskStubRepoProxy(repo, context, self.notifier)
    return repo
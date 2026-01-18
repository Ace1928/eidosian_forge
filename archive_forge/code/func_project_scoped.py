import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
@property
def project_scoped(self):
    """Return true if the auth token was scoped to a tenant (project).

        :returns: bool
        """
    return bool(self.project_id)
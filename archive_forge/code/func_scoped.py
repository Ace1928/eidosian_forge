import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
@property
def scoped(self):
    """Return true if the auth token was scoped.

        Returns true if scoped to a tenant(project) or domain,
        and contains a populated service catalog.

        This is deprecated, use project_scoped instead.

        :returns: bool
        """
    return self.project_scoped or self.domain_scoped or self.system_scoped
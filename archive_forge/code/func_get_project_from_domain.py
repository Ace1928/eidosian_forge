import abc
import keystone.conf
from keystone import exception
def get_project_from_domain(domain_ref):
    """Create a project ref from the provided domain ref."""
    project_ref = domain_ref.copy()
    project_ref['is_domain'] = True
    project_ref['domain_id'] = None
    project_ref['parent_id'] = None
    return project_ref
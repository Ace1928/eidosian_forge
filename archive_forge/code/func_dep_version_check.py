from .dependency_versions_table import deps
from .utils.versions import require_version, require_version_core
def dep_version_check(pkg, hint=None):
    require_version(deps[pkg], hint)
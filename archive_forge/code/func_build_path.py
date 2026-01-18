from breezy import osutils
def build_path(name):
    """Build and return a path in ssl_certs directory for name"""
    return osutils.pathjoin(base_dir, name)
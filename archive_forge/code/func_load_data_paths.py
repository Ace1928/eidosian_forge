import os, stat
def load_data_paths(*resource):
    """Returns an iterator which gives each directory named 'resource' in the
    application data search path. Information provided by earlier directories
    should take precedence over later ones."""
    resource = os.path.join(*resource)
    for data_dir in xdg_data_dirs:
        path = os.path.join(data_dir, resource)
        if os.path.exists(path):
            yield path
import os
import pkg_resources
def egg_info_dir(base_dir, dist_name):
    all = []
    for dir_extension in ['.'] + os.listdir(base_dir):
        full = os.path.join(base_dir, dir_extension, egg_name(dist_name) + '.egg-info')
        all.append(full)
        if os.path.exists(full):
            return full
    raise IOError('No egg-info directory found (looked in %s)' % ', '.join(all))
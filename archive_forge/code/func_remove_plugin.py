import os
import pkg_resources
def remove_plugin(egg_info_dir, plugin_name):
    """
    Remove the plugin to the given distribution (or spec), in
    .egg-info/paster_plugins.txt.  Raises ValueError if the
    plugin is not in the file.
    """
    fn = os.path.join(egg_info_dir, 'paster_plugins.txt')
    if not os.path.exists(fn):
        raise ValueError('Cannot remove plugin from %s; file does not exist' % fn)
    f = open(fn)
    lines = [l.strip() for l in f.readlines() if l.strip()]
    f.close()
    for line in lines:
        if line.lower() == plugin_name.lower():
            break
    else:
        raise ValueError('Plugin %s not found in file %s (from: %s)' % (plugin_name, fn, lines))
    lines.remove(line)
    print('writing', lines)
    f = open(fn, 'w')
    for line in lines:
        f.write(line)
        f.write('\n')
    f.close()
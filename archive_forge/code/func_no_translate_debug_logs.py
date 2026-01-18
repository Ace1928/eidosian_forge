import re
from hacking import core
@core.flake8ext
def no_translate_debug_logs(logical_line, filename):
    dirs = ['glance/api', 'glance/cmd', 'glance/common', 'glance/db', 'glance/domain', 'glance/image_cache', 'glance/quota', 'glance/store', 'glance/tests']
    if max([name in filename for name in dirs]):
        if logical_line.startswith('LOG.debug(_('):
            yield (0, "G319: Don't translate debug level logs")
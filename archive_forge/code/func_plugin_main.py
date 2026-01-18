import os
import re
import random
from gimpfu import *
def plugin_main(dirname, do_opaque, do_binary, do_alpha):
    if not dirname:
        pdb.gimp_message('No output directory selected, aborting')
        return
    if not os.path.isdir(dirname) or not os.access(dirname, os.W_OK):
        pdb.gimp_message('Invalid / non-writeable output directory, aborting')
        return
    tests = []
    tests.extend({0: ['OPAQUE', 'GRAY-OPAQUE'], 2: ['OPAQUE'], 3: ['GRAY-OPAQUE']}.get(do_opaque, []))
    tests.extend({0: ['BINARY', 'GRAY-BINARY'], 2: ['BINARY'], 3: ['GRAY-BINARY']}.get(do_binary, []))
    tests.extend({0: ['ALPHA', 'GRAY-ALPHA'], 2: ['ALPHA'], 3: ['GRAY-ALPHA']}.get(do_alpha, []))
    suite_cfg = dict(TESTSUITE_CONFIG)
    for testname, cfg in suite_cfg.items():
        if testname not in tests:
            continue
        pchars, inc, exc = cfg.pop('patterns')
        if not pchars:
            continue
        patterns = makepatterns(pchars, inc, exc)
        for alpha in cfg.pop('alpha', [255]):
            for layertype, exts in cfg.items():
                if not exts:
                    continue
                for p in patterns:
                    make_images(testname, p, alpha, layertype, exts, dirname)
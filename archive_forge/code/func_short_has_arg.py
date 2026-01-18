import os
def short_has_arg(opt, shortopts):
    for i in range(len(shortopts)):
        if opt == shortopts[i] != ':':
            return shortopts.startswith(':', i + 1)
    raise GetoptError(_('option -%s not recognized') % opt, opt)
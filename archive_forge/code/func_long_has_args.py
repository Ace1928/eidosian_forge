import os
def long_has_args(opt, longopts):
    possibilities = [o for o in longopts if o.startswith(opt)]
    if not possibilities:
        raise GetoptError(_('option --%s not recognized') % opt, opt)
    if opt in possibilities:
        return (False, opt)
    elif opt + '=' in possibilities:
        return (True, opt)
    if len(possibilities) > 1:
        raise GetoptError(_('option --%s not a unique prefix') % opt, opt)
    assert len(possibilities) == 1
    unique_match = possibilities[0]
    has_arg = unique_match.endswith('=')
    if has_arg:
        unique_match = unique_match[:-1]
    return (has_arg, unique_match)
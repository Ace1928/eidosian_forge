import os
def do_longs(opts, opt, longopts, args):
    try:
        i = opt.index('=')
    except ValueError:
        optarg = None
    else:
        opt, optarg = (opt[:i], opt[i + 1:])
    has_arg, opt = long_has_args(opt, longopts)
    if has_arg:
        if optarg is None:
            if not args:
                raise GetoptError(_('option --%s requires argument') % opt, opt)
            optarg, args = (args[0], args[1:])
    elif optarg is not None:
        raise GetoptError(_('option --%s must not have an argument') % opt, opt)
    opts.append(('--' + opt, optarg or ''))
    return (opts, args)
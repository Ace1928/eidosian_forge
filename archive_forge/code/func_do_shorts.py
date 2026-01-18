import os
def do_shorts(opts, optstring, shortopts, args):
    while optstring != '':
        opt, optstring = (optstring[0], optstring[1:])
        if short_has_arg(opt, shortopts):
            if optstring == '':
                if not args:
                    raise GetoptError(_('option -%s requires argument') % opt, opt)
                optstring, args = (args[0], args[1:])
            optarg, optstring = (optstring, '')
        else:
            optarg = ''
        opts.append(('-' + opt, optarg))
    return (opts, args)
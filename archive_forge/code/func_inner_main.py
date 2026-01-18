from __future__ import unicode_literals
import argparse
import io
import logging
import os
import sys
import cmakelang
from cmakelang import common
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.lint import basic_checker
from cmakelang.lint import lint_util
def inner_main():
    """Parse arguments, open files, start work."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, usage=USAGE_STRING)
    setup_argparse(argparser)
    try:
        import argcomplete
        argcomplete.autocomplete(argparser)
    except ImportError:
        pass
    args = argparser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    if args.dump_config:
        config_dict = __main__.get_config(os.getcwd(), args.config_files)
        __main__.dump_config(args, config_dict, sys.stdout)
        sys.exit(0)
    if args.outfile_path is None:
        args.outfile_path = '-'
    if '-' in args.infilepaths:
        assert len(args.infilepaths) == 1, 'You cannot mix stdin as an input with other input files'
    if args.outfile_path == '-':
        outfile = io.open(os.dup(sys.stdout.fileno()), mode='w', encoding='utf-8', newline='')
    else:
        outfile = io.open(args.outfile_path, 'w', encoding='utf-8', newline='')
    global_ctx = lint_util.GlobalContext(outfile)
    returncode = 0
    argdict = __main__.get_argdict(args)
    for infile_path in args.infilepaths:
        if infile_path == '-':
            config_dict = __main__.get_config(os.getcwd(), args.config_files)
        else:
            config_dict = __main__.get_config(infile_path, args.config_files)
        config_dict.update(argdict)
        cfg = configuration.Configuration(**config_dict)
        if infile_path == '-':
            infile_path = os.dup(sys.stdin.fileno())
        try:
            infile = io.open(infile_path, mode='r', encoding=cfg.encode.input_encoding, newline='')
        except (IOError, OSError):
            logger.error('Failed to open %s for read', infile_path)
            returncode = 1
            continue
        try:
            with infile:
                intext = infile.read()
        except UnicodeDecodeError:
            logger.error('Unable to read %s as %s', infile_path, cfg.encode.input_encoding)
            returncode = 1
            continue
        local_ctx = global_ctx.get_file_ctx(infile_path, cfg)
        process_file(cfg, local_ctx, intext)
        if not args.suppress_decorations:
            outfile.write('{}\n{}\n'.format(infile_path, '=' * len(infile_path)))
        local_ctx.writeout(outfile)
        if not args.suppress_decorations:
            outfile.write('\n')
        if local_ctx.has_lint():
            returncode = 1
    if not args.suppress_decorations:
        global_ctx.write_summary(outfile)
    outfile.close()
    return returncode
from __future__ import print_function
import sys
import getopt
from textwrap import dedent
from pygments import __version__, highlight
from pygments.util import ClassNotFound, OptionError, docstring_headline, \
from pygments.lexers import get_all_lexers, get_lexer_by_name, guess_lexer, \
from pygments.lexers.special import TextLexer
from pygments.formatters.latex import LatexEmbeddedLexer, LatexFormatter
from pygments.formatters import get_all_formatters, get_formatter_by_name, \
from pygments.formatters.terminal import TerminalFormatter
from pygments.filters import get_all_filters, find_filter_class
from pygments.styles import get_all_styles, get_style_by_name
def main_inner(popts, args, usage):
    opts = {}
    O_opts = []
    P_opts = []
    F_opts = []
    for opt, arg in popts:
        if opt == '-O':
            O_opts.append(arg)
        elif opt == '-P':
            P_opts.append(arg)
        elif opt == '-F':
            F_opts.append(arg)
        opts[opt] = arg
    if opts.pop('-h', None) is not None:
        print(usage)
        return 0
    if opts.pop('-V', None) is not None:
        print('Pygments version %s, (c) 2006-2017 by Georg Brandl.' % __version__)
        return 0
    L_opt = opts.pop('-L', None)
    if L_opt is not None:
        if opts:
            print(usage, file=sys.stderr)
            return 2
        main(['', '-V'])
        if not args:
            args = ['lexer', 'formatter', 'filter', 'style']
        for arg in args:
            _print_list(arg.rstrip('s'))
        return 0
    H_opt = opts.pop('-H', None)
    if H_opt is not None:
        if opts or len(args) != 2:
            print(usage, file=sys.stderr)
            return 2
        what, name = args
        if what not in ('lexer', 'formatter', 'filter'):
            print(usage, file=sys.stderr)
            return 2
        return _print_help(what, name)
    parsed_opts = _parse_options(O_opts)
    opts.pop('-O', None)
    for p_opt in P_opts:
        try:
            name, value = p_opt.split('=', 1)
        except ValueError:
            parsed_opts[p_opt] = True
        else:
            parsed_opts[name] = value
    opts.pop('-P', None)
    inencoding = parsed_opts.get('inencoding', parsed_opts.get('encoding'))
    outencoding = parsed_opts.get('outencoding', parsed_opts.get('encoding'))
    infn = opts.pop('-N', None)
    if infn is not None:
        lexer = find_lexer_class_for_filename(infn)
        if lexer is None:
            lexer = TextLexer
        print(lexer.aliases[0])
        return 0
    S_opt = opts.pop('-S', None)
    a_opt = opts.pop('-a', None)
    if S_opt is not None:
        f_opt = opts.pop('-f', None)
        if not f_opt:
            print(usage, file=sys.stderr)
            return 2
        if opts or args:
            print(usage, file=sys.stderr)
            return 2
        try:
            parsed_opts['style'] = S_opt
            fmter = get_formatter_by_name(f_opt, **parsed_opts)
        except ClassNotFound as err:
            print(err, file=sys.stderr)
            return 1
        print(fmter.get_style_defs(a_opt or ''))
        return 0
    if a_opt is not None:
        print(usage, file=sys.stderr)
        return 2
    F_opts = _parse_filters(F_opts)
    opts.pop('-F', None)
    allow_custom_lexer_formatter = False
    if opts.pop('-x', None) is not None:
        allow_custom_lexer_formatter = True
    lexer = None
    lexername = opts.pop('-l', None)
    if lexername:
        if allow_custom_lexer_formatter and '.py' in lexername:
            try:
                if ':' in lexername:
                    filename, name = lexername.rsplit(':', 1)
                    lexer = load_lexer_from_file(filename, name, **parsed_opts)
                else:
                    lexer = load_lexer_from_file(lexername, **parsed_opts)
            except ClassNotFound as err:
                print('Error:', err, file=sys.stderr)
                return 1
        else:
            try:
                lexer = get_lexer_by_name(lexername, **parsed_opts)
            except (OptionError, ClassNotFound) as err:
                print('Error:', err, file=sys.stderr)
                return 1
    code = None
    if args:
        if len(args) > 1:
            print(usage, file=sys.stderr)
            return 2
        if '-s' in opts:
            print('Error: -s option not usable when input file specified', file=sys.stderr)
            return 2
        infn = args[0]
        try:
            with open(infn, 'rb') as infp:
                code = infp.read()
        except Exception as err:
            print('Error: cannot read infile:', err, file=sys.stderr)
            return 1
        if not inencoding:
            code, inencoding = guess_decode(code)
        if not lexer:
            try:
                lexer = get_lexer_for_filename(infn, code, **parsed_opts)
            except ClassNotFound as err:
                if '-g' in opts:
                    try:
                        lexer = guess_lexer(code, **parsed_opts)
                    except ClassNotFound:
                        lexer = TextLexer(**parsed_opts)
                else:
                    print('Error:', err, file=sys.stderr)
                    return 1
            except OptionError as err:
                print('Error:', err, file=sys.stderr)
                return 1
    elif '-s' not in opts:
        if sys.version_info > (3,):
            code = sys.stdin.buffer.read()
        else:
            code = sys.stdin.read()
        if not inencoding:
            code, inencoding = guess_decode_from_terminal(code, sys.stdin)
        if not lexer:
            try:
                lexer = guess_lexer(code, **parsed_opts)
            except ClassNotFound:
                lexer = TextLexer(**parsed_opts)
    elif not lexer:
        print('Error: when using -s a lexer has to be selected with -l', file=sys.stderr)
        return 2
    for fname, fopts in F_opts:
        try:
            lexer.add_filter(fname, **fopts)
        except ClassNotFound as err:
            print('Error:', err, file=sys.stderr)
            return 1
    outfn = opts.pop('-o', None)
    fmter = opts.pop('-f', None)
    if fmter:
        if allow_custom_lexer_formatter and '.py' in fmter:
            try:
                if ':' in fmter:
                    file, fmtername = fmter.rsplit(':', 1)
                    fmter = load_formatter_from_file(file, fmtername, **parsed_opts)
                else:
                    fmter = load_formatter_from_file(fmter, **parsed_opts)
            except ClassNotFound as err:
                print('Error:', err, file=sys.stderr)
                return 1
        else:
            try:
                fmter = get_formatter_by_name(fmter, **parsed_opts)
            except (OptionError, ClassNotFound) as err:
                print('Error:', err, file=sys.stderr)
                return 1
    if outfn:
        if not fmter:
            try:
                fmter = get_formatter_for_filename(outfn, **parsed_opts)
            except (OptionError, ClassNotFound) as err:
                print('Error:', err, file=sys.stderr)
                return 1
        try:
            outfile = open(outfn, 'wb')
        except Exception as err:
            print('Error: cannot open outfile:', err, file=sys.stderr)
            return 1
    else:
        if not fmter:
            fmter = TerminalFormatter(**parsed_opts)
        if sys.version_info > (3,):
            outfile = sys.stdout.buffer
        else:
            outfile = sys.stdout
    if not outencoding:
        if outfn:
            fmter.encoding = inencoding
        else:
            fmter.encoding = terminal_encoding(sys.stdout)
    if not outfn and sys.platform in ('win32', 'cygwin') and (fmter.name in ('Terminal', 'Terminal256')):
        if sys.version_info > (3,):
            from pygments.util import UnclosingTextIOWrapper
            outfile = UnclosingTextIOWrapper(outfile, encoding=fmter.encoding)
            fmter.encoding = None
        try:
            import colorama.initialise
        except ImportError:
            pass
        else:
            outfile = colorama.initialise.wrap_stream(outfile, convert=None, strip=None, autoreset=False, wrap=True)
    escapeinside = parsed_opts.get('escapeinside', '')
    if len(escapeinside) == 2 and isinstance(fmter, LatexFormatter):
        left = escapeinside[0]
        right = escapeinside[1]
        lexer = LatexEmbeddedLexer(left, right, lexer)
    if '-s' not in opts:
        highlight(code, lexer, fmter, outfile)
        return 0
    else:
        try:
            while 1:
                if sys.version_info > (3,):
                    line = sys.stdin.buffer.readline()
                else:
                    line = sys.stdin.readline()
                if not line:
                    break
                if not inencoding:
                    line = guess_decode_from_terminal(line, sys.stdin)[0]
                highlight(line, lexer, fmter, outfile)
                if hasattr(outfile, 'flush'):
                    outfile.flush()
            return 0
        except KeyboardInterrupt:
            return 0
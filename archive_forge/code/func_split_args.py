from __future__ import (absolute_import, division, print_function)
def split_args(args):
    """
    Splits args on whitespace, but intelligently reassembles
    those that may have been split over a jinja2 block or quotes.

    When used in a remote module, we won't ever have to be concerned about
    jinja2 blocks, however this function is/will be used in the
    core portions as well before the args are templated.

    example input: a=b c="foo bar"
    example output: ['a=b', 'c="foo bar"']

    Basically this is a variation shlex that has some more intelligence for
    how Ansible needs to use it.
    """
    params = []
    args = args.strip()
    try:
        args = args.encode('utf-8')
        do_decode = True
    except UnicodeDecodeError:
        do_decode = False
    items = args.split('\n')
    quote_char = None
    inside_quotes = False
    print_depth = 0
    block_depth = 0
    comment_depth = 0
    for itemidx, item in enumerate(items):
        tokens = item.strip().split(' ')
        line_continuation = False
        for idx, token in enumerate(tokens):
            if token == '\\' and (not inside_quotes):
                line_continuation = True
                continue
            was_inside_quotes = inside_quotes
            quote_char = _get_quote_state(token, quote_char)
            inside_quotes = quote_char is not None
            appended = False
            if inside_quotes and (not was_inside_quotes):
                params.append(token)
                appended = True
            elif print_depth or block_depth or comment_depth or inside_quotes or was_inside_quotes:
                if idx == 0 and (not inside_quotes) and was_inside_quotes:
                    params[-1] = '%s%s' % (params[-1], token)
                elif len(tokens) > 1:
                    spacer = ''
                    if idx > 0:
                        spacer = ' '
                    params[-1] = '%s%s%s' % (params[-1], spacer, token)
                else:
                    spacer = ''
                    if not params[-1].endswith('\n') and idx == 0:
                        spacer = '\n'
                    params[-1] = '%s%s%s' % (params[-1], spacer, token)
                appended = True
            prev_print_depth = print_depth
            print_depth = _count_jinja2_blocks(token, print_depth, '{{', '}}')
            if print_depth != prev_print_depth and (not appended):
                params.append(token)
                appended = True
            prev_block_depth = block_depth
            block_depth = _count_jinja2_blocks(token, block_depth, '{%', '%}')
            if block_depth != prev_block_depth and (not appended):
                params.append(token)
                appended = True
            prev_comment_depth = comment_depth
            comment_depth = _count_jinja2_blocks(token, comment_depth, '{#', '#}')
            if comment_depth != prev_comment_depth and (not appended):
                params.append(token)
                appended = True
            if not (print_depth or block_depth or comment_depth) and (not inside_quotes) and (not appended) and (token != ''):
                params.append(token)
        if len(items) > 1 and itemidx != len(items) - 1 and (not line_continuation):
            if not params[-1].endswith('\n') or item == '':
                params[-1] += '\n'
        line_continuation = False
    if print_depth or block_depth or comment_depth or inside_quotes:
        raise Exception('error while splitting arguments, either an unbalanced jinja2 block or quotes')
    if do_decode:
        params = [x.decode('utf-8') for x in params]
    return params
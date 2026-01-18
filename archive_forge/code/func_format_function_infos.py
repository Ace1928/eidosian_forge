from subprocess import check_output
import os.path
from collections import defaultdict
import inspect
from functools import partial
import numba
from numba.core.registry import cpu_target
from all overloads.
def format_function_infos(fninfos):
    buf = StringIO()
    try:
        print = bind_file_to_print(buf)
        title_line = 'Lowering Listing'
        print(title_line)
        print('=' * len(title_line))
        print(description)
        commit = git_hash()

        def format_fname(fn):
            try:
                fname = '{0}.{1}'.format(fn.__module__, get_func_name(fn))
            except AttributeError:
                fname = repr(fn)
            return (fn, fname)
        for fn, fname in sorted(map(format_fname, fninfos), key=lambda x: x[1]):
            impinfos = fninfos[fn]
            header_line = '``{0}``'.format(fname)
            print(header_line)
            print('-' * len(header_line))
            print()
            formatted_sigs = map(lambda x: format_signature(x['sig']), impinfos)
            sorted_impinfos = sorted(zip(formatted_sigs, impinfos), key=lambda x: x[0])
            col_signatures = ['Signature']
            col_urls = ['Definition']
            for fmtsig, info in sorted_impinfos:
                impl = info['impl']
                filename = impl['filename']
                lines = impl['lines']
                fname = impl['name']
                source = '{0} lines {1}-{2}'.format(filename, *lines)
                link = github_url.format(commit=commit, path=filename, firstline=lines[0], lastline=lines[1])
                url = '``{0}`` `{1} <{2}>`_'.format(fname, source, link)
                col_signatures.append(fmtsig)
                col_urls.append(url)
            max_width_col_sig = max(map(len, col_signatures))
            max_width_col_url = max(map(len, col_urls))
            padding = 2
            width_col_sig = padding * 2 + max_width_col_sig
            width_col_url = padding * 2 + max_width_col_url
            line_format = '{{0:^{0}}}  {{1:^{1}}}'.format(width_col_sig, width_col_url)
            print(line_format.format('=' * width_col_sig, '=' * width_col_url))
            print(line_format.format(col_signatures[0], col_urls[0]))
            print(line_format.format('=' * width_col_sig, '=' * width_col_url))
            for sig, url in zip(col_signatures[1:], col_urls[1:]):
                print(line_format.format(sig, url))
            print(line_format.format('=' * width_col_sig, '=' * width_col_url))
            print()
        return buf.getvalue()
    finally:
        buf.close()
import sys
import os
import io
import pathlib
import re
import argparse
import zipfile
import json
import pickle
import pprint
import urllib.parse
from typing import (
import torch.utils.show_pickle
def get_model_info(path_or_file, title=None, extra_file_size_limit=DEFAULT_EXTRA_FILE_SIZE_LIMIT):
    """Get JSON-friendly information about a model.

    The result is suitable for being saved as model_info.json,
    or passed to burn_in_info.
    """
    if isinstance(path_or_file, os.PathLike):
        default_title = os.fspath(path_or_file)
        file_size = path_or_file.stat().st_size
    elif isinstance(path_or_file, str):
        default_title = path_or_file
        file_size = pathlib.Path(path_or_file).stat().st_size
    else:
        default_title = 'buffer'
        path_or_file.seek(0, io.SEEK_END)
        file_size = path_or_file.tell()
        path_or_file.seek(0)
    title = title or default_title
    with zipfile.ZipFile(path_or_file) as zf:
        path_prefix = None
        zip_files = []
        for zi in zf.infolist():
            prefix = re.sub('/.*', '', zi.filename)
            if path_prefix is None:
                path_prefix = prefix
            elif prefix != path_prefix:
                raise Exception(f'Mismatched prefixes: {path_prefix} != {prefix}')
            zip_files.append(dict(filename=zi.filename, compression=zi.compress_type, compressed_size=zi.compress_size, file_size=zi.file_size))
        assert path_prefix is not None
        version = zf.read(path_prefix + '/version').decode('utf-8').strip()

        def get_pickle(name):
            assert path_prefix is not None
            with zf.open(path_prefix + f'/{name}.pkl') as handle:
                raw = torch.utils.show_pickle.DumpUnpickler(handle, catch_invalid_utf8=True).load()
                return hierarchical_pickle(raw)
        model_data = get_pickle('data')
        constants = get_pickle('constants')
        interned_strings: Dict[str, int] = {}

        def ist(s):
            if s not in interned_strings:
                interned_strings[s] = len(interned_strings)
            return interned_strings[s]
        code_files = {}
        for zi in zf.infolist():
            if not zi.filename.endswith('.py'):
                continue
            with zf.open(zi) as handle:
                raw_code = handle.read()
            with zf.open(zi.filename + '.debug_pkl') as handle:
                raw_debug = handle.read()
            debug_info_t = pickle.loads(raw_debug)
            text_table = None
            if len(debug_info_t) == 3 and isinstance(debug_info_t[0], str) and (debug_info_t[0] == 'FORMAT_WITH_STRING_TABLE'):
                _, text_table, content = debug_info_t

                def parse_new_format(line):
                    num, ((text_indexes, fname_idx, offset), start, end), tag = line
                    text = ''.join((text_table[x] for x in text_indexes))
                    fname = text_table[fname_idx]
                    return (num, ((text, fname, offset), start, end), tag)
                debug_info_t = map(parse_new_format, content)
            debug_info = list(debug_info_t)
            if not debug_info:
                debug_info.append((0, (('', '', 0), 0, 0)))
            if debug_info[-1][0] != len(raw_code):
                debug_info.append((len(raw_code), (('', '', 0), 0, 0)))
            code_parts = []
            for di, di_next in zip(debug_info, debug_info[1:]):
                start, source_range, *_ = di
                end = di_next[0]
                assert end > start
                source, s_start, s_end = source_range
                s_text, s_file, s_line = source
                if len(s_text) != len(s_text.encode('utf-8')):
                    s_start = 0
                    s_end = 0
                text = raw_code[start:end]
                code_parts.append([text.decode('utf-8'), ist(s_file), s_line, ist(s_text), s_start, s_end])
            code_files[zi.filename] = code_parts
        extra_files_json_pattern = re.compile(re.escape(path_prefix) + '/extra/.*\\.json')
        extra_files_jsons = {}
        for zi in zf.infolist():
            if not extra_files_json_pattern.fullmatch(zi.filename):
                continue
            if zi.file_size > extra_file_size_limit:
                continue
            with zf.open(zi) as handle:
                try:
                    json_content = json.load(handle)
                    extra_files_jsons[zi.filename] = json_content
                except json.JSONDecodeError:
                    extra_files_jsons[zi.filename] = 'INVALID JSON'
        always_render_pickles = {'bytecode.pkl'}
        extra_pickles = {}
        for zi in zf.infolist():
            if not zi.filename.endswith('.pkl'):
                continue
            with zf.open(zi) as handle:
                obj = torch.utils.show_pickle.DumpUnpickler(handle, catch_invalid_utf8=True).load()
            buf = io.StringIO()
            pprint.pprint(obj, buf)
            contents = buf.getvalue()
            if os.path.basename(zi.filename) not in always_render_pickles and len(contents) > extra_file_size_limit:
                continue
            extra_pickles[zi.filename] = contents
    return {'model': dict(title=title, file_size=file_size, version=version, zip_files=zip_files, interned_strings=list(interned_strings), code_files=code_files, model_data=model_data, constants=constants, extra_files_jsons=extra_files_jsons, extra_pickles=extra_pickles)}
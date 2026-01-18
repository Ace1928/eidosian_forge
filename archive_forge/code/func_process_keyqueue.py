from __future__ import annotations
import re
import sys
import typing
from collections.abc import MutableMapping, Sequence
from urwid import str_util
import urwid.util  # isort: skip  # pylint: disable=wrong-import-position
def process_keyqueue(codes: Sequence[int], more_available: bool) -> tuple[list[str], Sequence[int]]:
    """
    codes -- list of key codes
    more_available -- if True then raise MoreInputRequired when in the
        middle of a character sequence (escape/utf8/wide) and caller
        will attempt to send more key codes on the next call.

    returns (list of input, list of remaining key codes).
    """
    code = codes[0]
    if 32 <= code <= 126:
        key = chr(code)
        return ([key], codes[1:])
    if code in _keyconv:
        return ([_keyconv[code]], codes[1:])
    if 0 < code < 27:
        return ([f'ctrl {ord('a') + code - 1:c}'], codes[1:])
    if 27 < code < 32:
        return ([f'ctrl {ord('A') + code - 1:c}'], codes[1:])
    em = str_util.get_byte_encoding()
    if em == 'wide' and code < 256 and within_double_byte(code.to_bytes(1, 'little'), 0, 0):
        if not codes[1:] and more_available:
            raise MoreInputRequired()
        if codes[1:] and codes[1] < 256:
            db = chr(code) + chr(codes[1])
            if within_double_byte(db, 0, 1):
                return ([db], codes[2:])
    if em == 'utf8' and 127 < code < 256:
        if code & 224 == 192:
            need_more = 1
        elif code & 240 == 224:
            need_more = 2
        elif code & 248 == 240:
            need_more = 3
        else:
            return ([f'<{code:d}>'], codes[1:])
        for i in range(1, need_more + 1):
            if len(codes) <= i:
                if more_available:
                    raise MoreInputRequired()
                return ([f'<{code:d}>'], codes[1:])
            k = codes[i]
            if k > 256 or k & 192 != 128:
                return ([f'<{code:d}>'], codes[1:])
        s = bytes(codes[:need_more + 1])
        try:
            return ([s.decode('utf-8')], codes[need_more + 1:])
        except UnicodeDecodeError:
            return ([f'<{code:d}>'], codes[1:])
    if 127 < code < 256:
        key = chr(code)
        return ([key], codes[1:])
    if code != 27:
        return ([f'<{code:d}>'], codes[1:])
    result = input_trie.get(codes[1:], more_available)
    if result is not None:
        result, remaining_codes = result
        return ([result], remaining_codes)
    if codes[1:]:
        run, remaining_codes = process_keyqueue(codes[1:], more_available)
        if urwid.util.is_mouse_event(run[0]):
            return (['esc', *run], remaining_codes)
        if run[0] == 'esc' or run[0].find('meta ') >= 0:
            return (['esc', *run], remaining_codes)
        return ([f'meta {run[0]}'] + run[1:], remaining_codes)
    return (['esc'], codes[1:])
import re
from json import scanner
def py_scanstring(s, end, strict=True, _b=BACKSLASH, _m=STRINGCHUNK.match):
    """Scan the string s for a JSON string. End is the index of the
    character in s after the quote that started the JSON string.
    Unescapes all valid JSON string escape sequences and raises ValueError
    on attempt to decode an invalid string. If strict is False then literal
    control characters are allowed in the string.

    Returns a tuple of the decoded string and the index of the character in s
    after the end quote."""
    chunks = []
    _append = chunks.append
    begin = end - 1
    while 1:
        chunk = _m(s, end)
        if chunk is None:
            raise JSONDecodeError('Unterminated string starting at', s, begin)
        end = chunk.end()
        content, terminator = chunk.groups()
        if content:
            _append(content)
        if terminator == '"':
            break
        elif terminator != '\\':
            if strict:
                msg = 'Invalid control character {0!r} at'.format(terminator)
                raise JSONDecodeError(msg, s, end)
            else:
                _append(terminator)
                continue
        try:
            esc = s[end]
        except IndexError:
            raise JSONDecodeError('Unterminated string starting at', s, begin) from None
        if esc != 'u':
            try:
                char = _b[esc]
            except KeyError:
                msg = 'Invalid \\escape: {0!r}'.format(esc)
                raise JSONDecodeError(msg, s, end)
            end += 1
        else:
            uni = _decode_uXXXX(s, end)
            end += 5
            if 55296 <= uni <= 56319 and s[end:end + 2] == '\\u':
                uni2 = _decode_uXXXX(s, end + 1)
                if 56320 <= uni2 <= 57343:
                    uni = 65536 + (uni - 55296 << 10 | uni2 - 56320)
                    end += 6
            char = chr(uni)
        _append(char)
    return (''.join(chunks), end)
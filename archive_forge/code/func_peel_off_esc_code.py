from typing import (
import re
from .termformatconstants import (
def peel_off_esc_code(s: str) -> Tuple[str, Optional[Token], str]:
    """Returns processed text, the next token, and unprocessed text

    >>> front, d, rest = peel_off_esc_code('some\x1b[2Astuff')
    >>> front, rest
    ('some', 'stuff')
    >>> d == {'numbers': [2], 'command': 'A', 'intermed': '', 'private': '', 'csi': '\\x1b[', 'seq': '\\x1b[2A'}
    True
    """
    p = '(?P<front>.*?)\n            (?P<seq>\n                (?P<csi>\n                    (?:[\x1b]\\[)\n                    |\n                    [' + '\x9b' + '])\n                (?P<private>)\n                (?P<numbers>\n                    (?:\\d+;)*\n                    (?:\\d+)?)\n                (?P<intermed>' + '[ -/]*)' + '\n                (?P<command>' + '[@-~]))' + '\n            (?P<rest>.*)'
    m1 = re.match(p, s, re.VERBOSE)
    m2 = re.match('(?P<front>.*?)(?P<seq>(?P<csi>\x1b)(?P<command>[@-_]))(?P<rest>.*)', s)
    m = None
    if m1 and m2:
        m = m1 if len(m1.groupdict()['front']) <= len(m2.groupdict()['front']) else m2
    elif m1:
        m = m1
    elif m2:
        m = m2
    else:
        m = None
    if m:
        d: Dict[str, Any] = m.groupdict()
        del d['front']
        del d['rest']
        if 'numbers' in d and all(d['numbers'].split(';')):
            d['numbers'] = [int(x) for x in d['numbers'].split(';')]
        return (m.groupdict()['front'], cast(Token, d), m.groupdict()['rest'])
    else:
        return (s, None, '')
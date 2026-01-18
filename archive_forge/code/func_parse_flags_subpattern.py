import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_flags_subpattern(source, info):
    """Parses a flags subpattern. It could be inline flags or a subpattern
    possibly with local flags. If it's a subpattern, then that's returned;
    if it's a inline flags, then None is returned.
    """
    flags_on, flags_off = parse_flags(source, info)
    if flags_off & GLOBAL_FLAGS:
        raise error('bad inline flags: cannot turn off global flag', source.string, source.pos)
    if flags_on & flags_off:
        raise error('bad inline flags: flag turned on and off', source.string, source.pos)
    new_global_flags = flags_on & ~info.global_flags & GLOBAL_FLAGS
    if new_global_flags:
        info.global_flags |= new_global_flags
        raise _UnscopedFlagSet(info.global_flags)
    flags_on &= ~GLOBAL_FLAGS
    if source.match(':'):
        return parse_subpattern(source, info, flags_on, flags_off)
    if source.match(')'):
        parse_positional_flags(source, info, flags_on, flags_off)
        return None
    raise error('unknown extension', source.string, source.pos)
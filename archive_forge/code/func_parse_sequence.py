import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_sequence(source, info):
    """Parses a sequence, eg. 'abc'."""
    sequence = [None]
    case_flags = make_case_flags(info)
    while True:
        saved_pos = source.pos
        ch = source.get()
        if ch in SPECIAL_CHARS:
            if ch in ')|':
                source.pos = saved_pos
                break
            elif ch == '\\':
                sequence.append(parse_escape(source, info, False))
            elif ch == '(':
                element = parse_paren(source, info)
                if element is None:
                    case_flags = make_case_flags(info)
                else:
                    sequence.append(element)
            elif ch == '.':
                if info.flags & DOTALL:
                    sequence.append(AnyAll())
                elif info.flags & WORD:
                    sequence.append(AnyU())
                else:
                    sequence.append(Any())
            elif ch == '[':
                sequence.append(parse_set(source, info))
            elif ch == '^':
                if info.flags & MULTILINE:
                    if info.flags & WORD:
                        sequence.append(StartOfLineU())
                    else:
                        sequence.append(StartOfLine())
                else:
                    sequence.append(StartOfString())
            elif ch == '$':
                if info.flags & MULTILINE:
                    if info.flags & WORD:
                        sequence.append(EndOfLineU())
                    else:
                        sequence.append(EndOfLine())
                elif info.flags & WORD:
                    sequence.append(EndOfStringLineU())
                else:
                    sequence.append(EndOfStringLine())
            elif ch in '?*+{':
                counts = parse_quantifier(source, info, ch)
                if counts:
                    apply_quantifier(source, info, counts, case_flags, ch, saved_pos, sequence)
                    sequence.append(None)
                else:
                    constraints = parse_fuzzy(source, info, ch, case_flags)
                    if constraints:
                        apply_constraint(source, info, constraints, case_flags, saved_pos, sequence)
                        sequence.append(None)
                    else:
                        sequence.append(Character(ord(ch), case_flags=case_flags))
            else:
                sequence.append(Character(ord(ch), case_flags=case_flags))
        else:
            sequence.append(Character(ord(ch), case_flags=case_flags))
    sequence = [item for item in sequence if item is not None]
    return Sequence(sequence)
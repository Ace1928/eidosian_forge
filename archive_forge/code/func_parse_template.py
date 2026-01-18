from ._constants import *
def parse_template(source, state):
    s = Tokenizer(source)
    sget = s.get
    groups = []
    literals = []
    literal = []
    lappend = literal.append

    def addgroup(index, pos):
        if index > state.groups:
            raise s.error('invalid group reference %d' % index, pos)
        if literal:
            literals.append(''.join(literal))
            del literal[:]
        groups.append((len(literals), index))
        literals.append(None)
    groupindex = state.groupindex
    while True:
        this = sget()
        if this is None:
            break
        if this[0] == '\\':
            c = this[1]
            if c == 'g':
                if not s.match('<'):
                    raise s.error('missing <')
                name = s.getuntil('>', 'group name')
                if name.isidentifier():
                    s.checkgroupname(name, 1, -1)
                    try:
                        index = groupindex[name]
                    except KeyError:
                        raise IndexError('unknown group name %r' % name) from None
                else:
                    try:
                        index = int(name)
                        if index < 0:
                            raise ValueError
                    except ValueError:
                        raise s.error('bad character in group name %r' % name, len(name) + 1) from None
                    if index >= MAXGROUPS:
                        raise s.error('invalid group reference %d' % index, len(name) + 1)
                    if not (name.isdecimal() and name.isascii()):
                        import warnings
                        warnings.warn('bad character in group name %s at position %d' % (repr(name) if s.istext else ascii(name), s.tell() - len(name) - 1), DeprecationWarning, stacklevel=5)
                addgroup(index, len(name) + 1)
            elif c == '0':
                if s.next in OCTDIGITS:
                    this += sget()
                    if s.next in OCTDIGITS:
                        this += sget()
                lappend(chr(int(this[1:], 8) & 255))
            elif c in DIGITS:
                isoctal = False
                if s.next in DIGITS:
                    this += sget()
                    if c in OCTDIGITS and this[2] in OCTDIGITS and (s.next in OCTDIGITS):
                        this += sget()
                        isoctal = True
                        c = int(this[1:], 8)
                        if c > 255:
                            raise s.error('octal escape value %s outside of range 0-0o377' % this, len(this))
                        lappend(chr(c))
                if not isoctal:
                    addgroup(int(this[1:]), len(this) - 1)
            else:
                try:
                    this = chr(ESCAPES[this][1])
                except KeyError:
                    if c in ASCIILETTERS:
                        raise s.error('bad escape %s' % this, len(this)) from None
                lappend(this)
        else:
            lappend(this)
    if literal:
        literals.append(''.join(literal))
    if not isinstance(source, str):
        literals = [None if s is None else s.encode('latin-1') for s in literals]
    return (groups, literals)
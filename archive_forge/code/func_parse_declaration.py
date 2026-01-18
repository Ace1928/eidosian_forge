import re
def parse_declaration(self, i):
    rawdata = self.rawdata
    j = i + 2
    assert rawdata[i:j] == '<!', 'unexpected call to parse_declaration'
    if rawdata[j:j + 1] == '>':
        return j + 1
    if rawdata[j:j + 1] in ('-', ''):
        return -1
    n = len(rawdata)
    if rawdata[j:j + 2] == '--':
        return self.parse_comment(i)
    elif rawdata[j] == '[':
        return self.parse_marked_section(i)
    else:
        decltype, j = self._scan_name(j, i)
    if j < 0:
        return j
    if decltype == 'doctype':
        self._decl_otherchars = ''
    while j < n:
        c = rawdata[j]
        if c == '>':
            data = rawdata[i + 2:j]
            if decltype == 'doctype':
                self.handle_decl(data)
            else:
                self.unknown_decl(data)
            return j + 1
        if c in '"\'':
            m = _declstringlit_match(rawdata, j)
            if not m:
                return -1
            j = m.end()
        elif c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            name, j = self._scan_name(j, i)
        elif c in self._decl_otherchars:
            j = j + 1
        elif c == '[':
            if decltype == 'doctype':
                j = self._parse_doctype_subset(j + 1, i)
            elif decltype in {'attlist', 'linktype', 'link', 'element'}:
                raise AssertionError("unsupported '[' char in %s declaration" % decltype)
            else:
                raise AssertionError("unexpected '[' char in declaration")
        else:
            raise AssertionError('unexpected %r char in declaration' % rawdata[j])
        if j < 0:
            return j
    return -1
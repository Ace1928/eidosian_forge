def dump_char(self, code):
    if 0 <= code <= 255:
        return repr(chr(code))
    else:
        return 'chr(%d)' % code
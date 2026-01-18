import sys
def readLine(self, line):
    self._lineNo = self._lineNo + 1
    stripped = line.lstrip()
    if len(stripped) == 0:
        if self._mode == PLAIN:
            self.endPara()
        else:
            self._buf.append(line)
    elif line[0] == '.':
        self.endPara()
        words = stripped[1:].split()
        cmd, args = (words[0], words[1:])
        if hasattr(self.__class__, cmd):
            try:
                getattr(self, cmd)(*args)
            except TypeError as err:
                sys.stderr.write('Parser method: %s(*%s) %s at line %d\n' % (cmd, args, err, self._lineNo))
                raise
        else:
            self.endPara()
            words = stripped.split(' ', 1)
            assert len(words) == 2, 'Style %s but no data at line %d' % (words[0], self._lineNo)
            styletag, data = words
            self._style = styletag[1:]
            self._buf.append(data)
    else:
        self._buf.append(line)
from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def dumphex(self, data):
    linelength = 16
    hexlinelength = linelength * 2
    chunksize = 8
    for i in range(0, len(data), linelength):
        hexline = hexStr(data[i:i + linelength])
        line = ''
        white = ''
        for j in range(0, hexlinelength, chunksize):
            line = line + white + hexline[j:j + chunksize]
            white = ' '
        self._writeraw(line)
        self.newline()
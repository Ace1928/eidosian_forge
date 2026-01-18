import os
def loadfont(self, fontname):
    filename = AFMDIR + os.sep + fontname + '.afm'
    print('cache loading', filename)
    assert os.path.exists(filename)
    widths = parseAFMfile(filename)
    self.__widtharrays[fontname] = widths
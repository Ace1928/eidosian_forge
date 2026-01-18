from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_findfont(self):
    name = self.pop()
    font = self.dictstack[0]['FontDirectory'].value[name.value]
    self.push(font)
from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_def(self):
    obj = self.pop()
    name = self.pop()
    self.dictstack[-1][name.value] = obj
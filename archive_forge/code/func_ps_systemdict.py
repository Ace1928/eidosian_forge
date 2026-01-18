from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_systemdict(self):
    self.push(ps_dict(self.dictstack[0]))
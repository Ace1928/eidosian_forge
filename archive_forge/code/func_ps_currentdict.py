from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_currentdict(self):
    self.push(ps_dict(self.dictstack[-1]))
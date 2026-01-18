from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_count(self):
    self.push(ps_integer(len(self.stack)))
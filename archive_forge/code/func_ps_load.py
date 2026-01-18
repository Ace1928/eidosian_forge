from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_load(self):
    name = self.pop()
    self.push(self.resolve_name(name.value))
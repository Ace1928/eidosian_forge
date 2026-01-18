from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_string(self):
    num = self.pop('integertype').value
    self.push(ps_string('\x00' * num))
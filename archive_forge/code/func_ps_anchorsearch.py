from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_anchorsearch(self):
    seek = self.pop('stringtype')
    s = self.pop('stringtype')
    seeklen = len(seek.value)
    if s.value[:seeklen] == seek.value:
        self.push(ps_string(s.value[seeklen:]))
        self.push(seek)
        self.push(ps_boolean(1))
    else:
        self.push(s)
        self.push(ps_boolean(0))
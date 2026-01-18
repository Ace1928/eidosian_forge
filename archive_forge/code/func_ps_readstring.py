from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_readstring(self, ps_boolean=ps_boolean, len=len):
    s = self.pop('stringtype')
    oldstr = s.value
    f = self.pop('filetype')
    f.value.pos = f.value.pos + 1
    newstr = f.value.read(len(oldstr))
    s.value = newstr
    self.push(s)
    self.push(ps_boolean(len(oldstr) == len(newstr)))
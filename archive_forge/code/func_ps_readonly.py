from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_readonly(self):
    obj = self.pop()
    if obj.access < 1:
        obj.access = 1
    self.push(obj)
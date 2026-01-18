from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_store(self):
    value = self.pop()
    key = self.pop()
    name = key.value
    for i in range(len(self.dictstack) - 1, -1, -1):
        if name in self.dictstack[i]:
            self.dictstack[i][name] = value
            break
    self.dictstack[-1][name] = value
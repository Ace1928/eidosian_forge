from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_cleartomark(self):
    obj = self.pop()
    while obj != self.mark:
        obj = self.pop()
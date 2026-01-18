from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_closefile(self):
    f = self.pop('filetype').value
    f.skipwhite()
    f.stopeexec()
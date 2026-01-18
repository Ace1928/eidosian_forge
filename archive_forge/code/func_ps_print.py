from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_print(self):
    str = self.pop('stringtype')
    print('PS output --->', str.value)
from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_array(self):
    num = self.pop('integertype')
    array = ps_array([None] * num.value)
    self.push(array)
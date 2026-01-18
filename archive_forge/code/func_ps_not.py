from fontTools.encodings.StandardEncoding import StandardEncoding
def ps_not(self):
    obj = self.pop('booleantype', 'integertype')
    if obj.type == 'booleantype':
        self.push(ps_boolean(not obj.value))
    else:
        self.push(ps_integer(~obj.value))
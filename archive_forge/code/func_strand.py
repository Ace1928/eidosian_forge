def strand(self, arrow):
    sign = self.sign()
    if arrow not in self:
        return None
    elif arrow == self.over and sign == 'RH' or (arrow == self.under and sign == 'LH'):
        return 'X'
    else:
        return 'Y'
import time, calendar
def gotonext(self):
    """Skip white space and extract comments."""
    wslist = []
    while self.pos < len(self.field):
        if self.field[self.pos] in self.LWS + '\n\r':
            if self.field[self.pos] not in '\n\r':
                wslist.append(self.field[self.pos])
            self.pos += 1
        elif self.field[self.pos] == '(':
            self.commentlist.append(self.getcomment())
        else:
            break
    return EMPTYSTRING.join(wslist)
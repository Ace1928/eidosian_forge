import time, calendar
def getrouteaddr(self):
    """Parse a route address (Return-path value).

        This method just skips all the route stuff and returns the addrspec.
        """
    if self.field[self.pos] != '<':
        return
    expectroute = False
    self.pos += 1
    self.gotonext()
    adlist = ''
    while self.pos < len(self.field):
        if expectroute:
            self.getdomain()
            expectroute = False
        elif self.field[self.pos] == '>':
            self.pos += 1
            break
        elif self.field[self.pos] == '@':
            self.pos += 1
            expectroute = True
        elif self.field[self.pos] == ':':
            self.pos += 1
        else:
            adlist = self.getaddrspec()
            self.pos += 1
            break
        self.gotonext()
    return adlist
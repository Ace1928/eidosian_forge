import time, calendar
def getcomment(self):
    """Get a parenthesis-delimited fragment from self's field."""
    return self.getdelimited('(', ')\r', True)
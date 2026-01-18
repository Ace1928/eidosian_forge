import logging
def italics(self, s):
    if s:
        self.start_italics()
        self.doc.write(s)
        self.end_italics()
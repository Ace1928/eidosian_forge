import logging
def li(self, s):
    if s:
        self.start_li()
        self.doc.writeln(s)
        self.end_li()
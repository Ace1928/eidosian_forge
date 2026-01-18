import logging
def table_of_contents(self, title=None, depth=None):
    self.doc.write('.. contents:: ')
    if title is not None:
        self.doc.writeln(title)
    if depth is not None:
        self.doc.writeln('   :depth: %s' % depth)
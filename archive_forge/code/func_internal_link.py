import logging
def internal_link(self, title, page):
    if self.doc.target == 'html':
        self.doc.write(f':doc:`{title} <{page}>`')
    else:
        self.doc.write(title)
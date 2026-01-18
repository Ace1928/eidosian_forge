import logging
def hidden_tocitem(self, item):
    if self.doc.target == 'html':
        self.tocitem(item)
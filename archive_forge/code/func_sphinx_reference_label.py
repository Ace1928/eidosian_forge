import logging
def sphinx_reference_label(self, label, text=None):
    if text is None:
        text = label
    if self.doc.target == 'html':
        self.doc.write(f':ref:`{text} <{label}>`')
    else:
        self.doc.write(text)
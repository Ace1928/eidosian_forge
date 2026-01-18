import logging
def start_important(self, attrs=None):
    self.new_paragraph()
    self.doc.write('.. warning::')
    self.indent()
    self.new_paragraph()
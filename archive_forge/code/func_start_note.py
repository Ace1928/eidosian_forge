import logging
def start_note(self, attrs=None):
    self.new_paragraph()
    self.doc.write('.. note::')
    self.indent()
    self.new_paragraph()
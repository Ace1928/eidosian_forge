import logging
def start_codeblock(self, attrs=None):
    self.doc.write('::')
    self.indent()
    self.new_paragraph()
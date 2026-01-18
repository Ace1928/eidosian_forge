import logging
def start_sphinx_py_attr(self, attr_name):
    self.new_paragraph()
    self.doc.write('.. py:attribute:: %s' % attr_name)
    self.indent()
    self.new_paragraph()
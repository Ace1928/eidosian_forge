import logging
def start_sphinx_py_class(self, class_name):
    self.new_paragraph()
    self.doc.write('.. py:class:: %s' % class_name)
    self.indent()
    self.new_paragraph()
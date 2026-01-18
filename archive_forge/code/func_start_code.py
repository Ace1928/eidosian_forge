import logging
def start_code(self, attrs=None):
    self.doc.do_translation = True
    self.add_trailing_space_to_previous_write()
    self._start_inline('``')
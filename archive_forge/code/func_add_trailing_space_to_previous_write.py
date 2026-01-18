import logging
def add_trailing_space_to_previous_write(self):
    last_write = self.doc.pop_write()
    if last_write is None:
        last_write = ''
    if last_write != '' and last_write[-1] != ' ':
        last_write += ' '
    self.doc.push_write(last_write)
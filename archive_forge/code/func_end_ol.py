import logging
def end_ol(self):
    self.list_depth -= 1
    if self.list_depth != 0:
        self.dedent()
    self.new_paragraph()
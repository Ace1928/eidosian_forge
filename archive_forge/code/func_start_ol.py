import logging
def start_ol(self, attrs=None):
    if self.list_depth != 0:
        self.indent()
    self.list_depth += 1
    self.new_paragraph()
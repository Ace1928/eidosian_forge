def error_no_section_name(self, line):
    raise self.parse_exc('Empty section name', self.lineno, line)
def error_no_section_end_bracket(self, line):
    raise self.parse_exc('Invalid section (must end with ])', self.lineno, line)
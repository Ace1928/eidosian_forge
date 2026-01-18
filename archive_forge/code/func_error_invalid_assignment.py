def error_invalid_assignment(self, line):
    raise self.parse_exc("No ':' or '=' found in assignment", self.lineno, line)
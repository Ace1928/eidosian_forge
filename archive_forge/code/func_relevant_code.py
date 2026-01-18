def relevant_code(self):
    """Extracts and returns the span of the original code represented by
        this Origin. Example: ``x1``."""
    return self.code[self.start:self.end]
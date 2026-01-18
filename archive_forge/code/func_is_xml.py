def is_xml(self):
    """Returns True if this media type is XML-based.

        Note this does not consider text/html to be XML, but
        application/xhtml+xml is.
        """
    return self.minor == 'xml' or self.minor.endswith('+xml')
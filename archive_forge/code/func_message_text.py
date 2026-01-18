def message_text(self):
    u"""
        Format the above text with the name and minimum version required.
        """
    return message_unformatted % (self.name, self.version)
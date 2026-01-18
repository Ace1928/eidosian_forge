from typing import Optional
def parseLine(self, line):
    """
        Override this.

        By default, this will split the line on whitespace and call
        self.parseFields (catching any errors).
        """
    try:
        self.parseFields(*line.split())
    except ValueError:
        raise InvalidInetdConfError('Invalid line: ' + repr(line))
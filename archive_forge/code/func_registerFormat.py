from reportlab.rl_config import register_reset
def registerFormat(self, format, func):
    """Registers a new formatting function.  The funtion
        must take a number as argument and return a string;
        fmt is a short menmonic string used to access it."""
    self._formatters[format] = func
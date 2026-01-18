import sys
def formatwarning(message, category, filename, lineno, line=None):
    """Function to format a warning the standard way."""
    msg = WarningMessage(message, category, filename, lineno, None, line)
    return _formatwarnmsg_impl(msg)
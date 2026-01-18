import logging
def enableTrace(traceable, handler=logging.StreamHandler()):
    """
    turn on/off the traceability.

    traceable: boolean value. if set True, traceability is enabled.
    """
    global _traceEnabled
    _traceEnabled = traceable
    if traceable:
        _logger.addHandler(handler)
        _logger.setLevel(logging.DEBUG)
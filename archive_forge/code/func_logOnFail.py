import traceback, logging
from OpenGL._configflags import ERROR_LOGGING, FULL_LOGGING
def logOnFail(function, log):
    """Produce possible log-wrapped version of function

    function -- callable object to be wrapped
    log -- the log to which to log information
    
    Uses ERROR_LOGGING and FULL_LOGGING
    to determine whether/how to wrap the function.
    """
    if ERROR_LOGGING or FULL_LOGGING:
        if FULL_LOGGING:
            loggedFunction = _FullLoggedFunction(function, log)
        else:
            loggedFunction = _ErrorLoggedFunction(function, log)
        return loggedFunction
    else:
        return function
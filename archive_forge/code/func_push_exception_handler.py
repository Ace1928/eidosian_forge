import logging
import sys
def push_exception_handler(self, handler=None, reraise_exceptions=False):
    """ Push a new exception handler into the stack. Making it the
        current exception handler.

        Parameters
        ----------
        handler : callable(event) or None
            A callable to handle an event, in the context of
            an exception. If None, the exceptions will be logged.
        reraise_exceptions : boolean
            Whether to reraise the exception.
        """
    self.handlers.append(ObserverExceptionHandler(handler=handler, reraise_exceptions=reraise_exceptions))
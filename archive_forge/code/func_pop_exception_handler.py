import logging
import sys
def pop_exception_handler(self):
    """ Pop the current exception handler from the stack.

        Raises
        ------
        IndexError
            If there are no handlers to pop.
        """
    return self.handlers.pop()
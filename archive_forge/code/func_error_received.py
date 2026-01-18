def error_received(self, exc):
    """Called when a send or receive operation raises an OSError.

        (Other than BlockingIOError or InterruptedError.)
        """
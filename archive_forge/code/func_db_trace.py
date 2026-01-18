important invariant is that the parts on the stack are themselves in
def db_trace(self, msg):
    """Useful for understanding/debugging the algorithms.  Not
        generally activated in end-user code."""
    if self.debug:
        raise RuntimeError
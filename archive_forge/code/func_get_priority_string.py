from docutils import languages, ApplicationError, TransformSpec
def get_priority_string(self, priority):
    """
        Return a string, `priority` combined with `self.serialno`.

        This ensures FIFO order on transforms with identical priority.
        """
    self.serialno += 1
    return '%03d-%03d' % (priority, self.serialno)
import os
def random_all():
    """A function where we ignore the output of ALL examples.

    Examples:

      # all-random

      This mark tells the testing machinery that all subsequent examples should
      be treated as random (ignoring their output).  They are still executed,
      so if a they raise an error, it will be detected as such, but their
      output is completely ignored.

      >>> 1+3
      junk goes here...

      >>> 1+3
      klasdfj;

      >>> 1+2
      again,  anything goes
      blah...
    """
    pass
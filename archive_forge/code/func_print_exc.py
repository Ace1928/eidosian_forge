import gc
import itertools
import sys
import time
def print_exc(self, file=None):
    """Helper to print a traceback from the timed code.

        Typical use:

            t = Timer(...)       # outside the try/except
            try:
                t.timeit(...)    # or t.repeat(...)
            except:
                t.print_exc()

        The advantage over the standard traceback is that source lines
        in the compiled template will be displayed.

        The optional file argument directs where the traceback is
        sent; it defaults to sys.stderr.
        """
    import linecache, traceback
    if self.src is not None:
        linecache.cache[dummy_src_name] = (len(self.src), None, self.src.split('\n'), dummy_src_name)
    traceback.print_exc(file=file)
import doctest
import re
import decimal
def formatted_compare_numeric(self, want, got, optionflags):
    """
        Performs comparison of compare_numeric and returns formatted
        text.

        Only supposed to be used if comparison failed.
        """
    status, data = self.compare_numeric(want, got, optionflags)
    return self.format_compare_numeric_result(status, data)
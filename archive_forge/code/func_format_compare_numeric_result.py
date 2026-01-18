import doctest
import re
import decimal
def format_compare_numeric_result(self, status, data):
    """
        Formats a nice text from the result of compare_numeric.
        """
    if status == 'COUNT':
        return 'Expected %d numbers but got %d numbers.' % data
    elif status == 'TEXT':
        return 'Text between numbers differs: Expected "%s" but got "%s" at position %d' % data
    elif status == 'TYPE':
        is_interval_want, number_got = data
        if is_interval_want:
            k = 'interval'
        else:
            k = 'number'
        return 'Expected %s, but got %s.' % (k, number_got)
    elif status == 'NUMERIC':
        rows, max_diff = data
        result = 'Numbers differed by %s\n' % max_diff
        for number_want, number_got, failed, diff in rows:
            if result:
                result += '\n'
            result += 'Expected     : %s\n' % number_want
            result += 'Got          : %s\n' % number_got
            if failed:
                result += 'Difference (FAILURE): %s\n' % diff
            else:
                result += 'Difference          : %s\n' % diff
        return result
    else:
        raise Exception('Internal error in OutputChecker.')
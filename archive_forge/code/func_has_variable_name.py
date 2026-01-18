import sys
import six
def has_variable_name(s):
    """
    Variable name before [
    @param s:
    """
    if s.find('[') > 0:
        return True
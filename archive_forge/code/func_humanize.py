import re
import unicodedata
def humanize(word):
    """
    Capitalize the first word and turn underscores into spaces and strip a
    trailing ``"_id"``, if any. Like :func:`titleize`, this is meant for
    creating pretty output.

    Examples::

        >>> humanize("employee_salary")
        'Employee salary'
        >>> humanize("author_id")
        'Author'

    """
    word = re.sub('_id$', '', word)
    word = word.replace('_', ' ')
    word = re.sub('(?i)([a-z\\d]*)', lambda m: m.group(1).lower(), word)
    word = re.sub('^\\w', lambda m: m.group(0).upper(), word)
    return word
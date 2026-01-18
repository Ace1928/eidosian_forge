from collections import defaultdict
from ..core import Store
@classmethod
def extract_keywords(cls, line, items):
    """
        Given the keyword string, parse a dictionary of options.
        """
    unprocessed = list(reversed(line.split('=')))
    while unprocessed:
        chunk = unprocessed.pop()
        key = None
        if chunk.strip() in cls.allowed:
            key = chunk.strip()
        else:
            raise SyntaxError(f'Invalid keyword: {chunk.strip()}')
        value = unprocessed.pop().strip()
        if len(unprocessed) != 0:
            for option in cls.allowed:
                if value.endswith(option):
                    value = value[:-len(option)].strip()
                    unprocessed.append(option)
                    break
            else:
                raise SyntaxError(f'Invalid keyword: {value.split()[-1]}')
        keyword = f'{key}={value}'
        try:
            items.update(eval(f'dict({keyword})'))
        except Exception:
            raise SyntaxError(f'Could not evaluate keyword: {keyword}') from None
    return items
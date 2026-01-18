import logging
import re
def replace_carets(comp, loose):
    return ' '.join([replace_caret(c, loose) for c in re.split('\\s+', comp.strip())])
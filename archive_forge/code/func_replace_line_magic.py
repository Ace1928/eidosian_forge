import ast
from nbconvert.preprocessors import Preprocessor
def replace_line_magic(source, magic, template='{line}'):
    """
    Given a cell's source, replace line magics using a formatting
    template, where {line} is the string that follows the magic.
    """
    filtered = []
    for line in source.splitlines():
        if line.strip().startswith(magic):
            substitution = template.format(line=line.replace(magic, ''))
            filtered.append(substitution)
        else:
            filtered.append(line)
    return '\n'.join(filtered)
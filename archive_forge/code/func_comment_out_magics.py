import ast
from nbconvert.preprocessors import Preprocessor
def comment_out_magics(source):
    """
    Utility used to make sure AST parser does not choke on unrecognized
    magics.
    """
    filtered = []
    for line in source.splitlines():
        if line.strip().startswith('%'):
            filtered.append('# ' + line)
        else:
            filtered.append(line)
    return '\n'.join(filtered)
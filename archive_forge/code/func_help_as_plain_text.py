import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def help_as_plain_text(text):
    """Minimal converter of reStructuredText to plain text."""
    import re
    text = re.sub('(?m)^\\s*::\\n\\s*$', '', text)
    lines = text.splitlines()
    result = []
    for line in lines:
        if line.startswith(':'):
            line = line[1:]
        elif line.endswith('::'):
            line = line[:-1]
        line = re.sub(':doc:`(.+?)-help`', '``brz help \\1``', line)
        result.append(line)
    return '\n'.join(result) + '\n'
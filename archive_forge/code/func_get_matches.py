from .. import osutils
def get_matches(self, line):
    """Return the lines which match the line in right."""
    try:
        return self._matching_lines[line]
    except KeyError:
        return None
from .. import osutils
def extend_lines(self, lines, index):
    """Add more lines to the left-lines list.

        :param lines: A list of lines to add
        :param index: A True/False for each node to define if it should be
            indexed.
        """
    self._update_matching_lines(lines, index)
    self.lines.extend(lines)
    endpoint = self.endpoint
    for line in lines:
        endpoint += len(line)
        self.line_offsets.append(endpoint)
    if len(self.line_offsets) != len(self.lines):
        raise AssertionError('Somehow the line offset indicator got out of sync with the line counter.')
    self.endpoint = endpoint
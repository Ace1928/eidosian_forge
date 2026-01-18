import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def parse_targets(self, source):
    """
        Fetch and parse configuration statements that required for
        defining the targeted CPU features, statements should be declared
        in the top of source in between **C** comment and start
        with a special mark **@targets**.

        Configuration statements are sort of keywords representing
        CPU features names, group of statements and policies, combined
        together to determine the required optimization.

        Parameters
        ----------
        source : str
            the path of **C** source file.

        Returns
        -------
        - bool, True if group has the 'baseline' option
        - list, list of CPU features
        - list, list of extra compiler flags
        """
    self.dist_log("looking for '@targets' inside -> ", source)
    with open(source) as fd:
        tokens = ''
        max_to_reach = 1000
        start_with = '@targets'
        start_pos = -1
        end_with = '*/'
        end_pos = -1
        for current_line, line in enumerate(fd):
            if current_line == max_to_reach:
                self.dist_fatal('reached the max of lines')
                break
            if start_pos == -1:
                start_pos = line.find(start_with)
                if start_pos == -1:
                    continue
                start_pos += len(start_with)
            tokens += line
            end_pos = line.find(end_with)
            if end_pos != -1:
                end_pos += len(tokens) - len(line)
                break
    if start_pos == -1:
        self.dist_fatal("expected to find '%s' within a C comment" % start_with)
    if end_pos == -1:
        self.dist_fatal("expected to end with '%s'" % end_with)
    tokens = tokens[start_pos:end_pos]
    return self._parse_target_tokens(tokens)